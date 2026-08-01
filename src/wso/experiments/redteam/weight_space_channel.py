from __future__ import annotations

import struct
import tempfile
from pathlib import Path

import torch


class WeightSpaceCovertChannel:
    """
    Bounded red-team fixture: LSB steganography in FP32 mantissas.

    Purpose
    -------
    This is a controlled research fixture intended to demonstrate one class of
    attack against weight-space channels in PyTorch models. It is meant to help
    evaluate defensive telemetry systems by providing a concrete local example
    of hidden data transport via model weights.

    Important limitations
    ---------------------
    - This modifies the actual floating-point values. Any tensor change also
      changes serialized model bytes and is not invisible to hashes, signing, or
      strong provenance controls.
    - The magic header ("NAXX") is a detectable artifact if a defender scans the
      LSB stream of FP32 weights.
    - Survival is not guaranteed across quantization, dtype conversion,
      fine-tuning, pruning, optimizer steps, or some serialization and inference
      runtimes.
    - The fixture operates on tensors returned by state_dict() and does not
      preserve Parameter metadata such as requires_grad.

    Framing
    -------
    Payload is prefixed with:
        b"NAXX" (4 bytes magic) + little-endian uint32 length (4 bytes)
    Total header overhead = 8 bytes.
    Usable payload capacity in a tensor with N elements is approximately
    (N // 8) - 8 bytes.
    """

    MAGIC: bytes = b"NAXX"
    HEADER_SIZE: int = 8
    MAX_PAYLOAD_BYTES: int = (1 << 32) - 1

    def _encode_payload(self, payload: bytes) -> torch.Tensor:
        """Frame payload with magic + little-endian length, then expand to bits."""
        if not isinstance(payload, bytes):
            raise TypeError(f"payload must be bytes. Got {type(payload).__name__}")
        if len(payload) > self.MAX_PAYLOAD_BYTES:
            raise ValueError(
                f"payload is too large: {len(payload)} bytes exceeds "
                f"{self.MAX_PAYLOAD_BYTES} bytes"
            )

        header = self.MAGIC + struct.pack("<I", len(payload))
        full_data = header + payload

        bit_list: list[int] = []
        for byte in full_data:
            for bit_index in range(8):
                bit_list.append((byte >> bit_index) & 1)
        return torch.tensor(bit_list, dtype=torch.uint8)

    def _decode_bits(self, bits: torch.Tensor) -> bytes:
        """Reconstruct bytes from bit tensor, LSB-first within each byte."""
        bits_list = bits.tolist()
        byte_list = bytearray()
        for offset in range(0, len(bits_list), 8):
            byte = 0
            for bit_index in range(8):
                if offset + bit_index < len(bits_list):
                    byte |= bits_list[offset + bit_index] << bit_index
            byte_list.append(byte)
        return bytes(byte_list)

    def inject(
        self,
        state_dict: dict[str, torch.Tensor],
        layer_name: str,
        payload_bytes: bytes,
    ) -> None:
        """
        Inject framed payload into the LSB of the mantissa of the target tensor.

        Operates on a detached contiguous clone and writes back through copy_()
        so the original tensor layout is preserved.
        """
        if layer_name not in state_dict:
            raise KeyError(f"Layer '{layer_name}' not found in state_dict")

        param = state_dict[layer_name]
        if not isinstance(param, torch.Tensor) or param.dtype != torch.float32:
            raise TypeError(
                f"Parameter '{layer_name}' must be float32. Got {param.dtype}"
            )

        payload_bits = self._encode_payload(payload_bytes)
        n_bits = payload_bits.numel()
        if n_bits > param.numel():
            raise ValueError(
                f"Payload requires {n_bits} bits "
                f"but layer '{layer_name}' only has {param.numel()} elements"
            )

        flat = param.detach().clone().contiguous().view(-1)
        int_view = flat.view(torch.int32)
        device = flat.device
        clear_mask = torch.tensor(-2, dtype=torch.int32, device=device)

        int_view[:n_bits] &= clear_mask
        int_view[:n_bits] |= payload_bits.to(dtype=torch.int32, device=device)

        with torch.no_grad():
            param.copy_(flat.view(param.shape))

    def extract(
        self,
        state_dict: dict[str, torch.Tensor],
        layer_name: str,
    ) -> bytes:
        """
        Read the framed LSB stream from the target tensor and return the payload.

        The fixture expects the frame at the start of the LSB stream. This keeps
        extraction deterministic for tests and avoids scanning arbitrary offsets.
        """
        if layer_name not in state_dict:
            raise KeyError(f"Layer '{layer_name}' not found in state_dict")

        param = state_dict[layer_name]
        if not isinstance(param, torch.Tensor) or param.dtype != torch.float32:
            raise TypeError(
                f"Parameter '{layer_name}' must be float32. Got {param.dtype}"
            )

        flat = param.detach().contiguous().view(-1)
        int_view = flat.view(torch.int32)
        lsbs = (int_view & 1).cpu().to(torch.uint8)

        header_bits = self.HEADER_SIZE * 8
        if lsbs.numel() < header_bits:
            raise ValueError(
                f"Layer '{layer_name}' is too small for a payload header. "
                f"Need {header_bits} elements, got {lsbs.numel()}."
            )

        header = self._decode_bits(lsbs[:header_bits])
        if header[: len(self.MAGIC)] != self.MAGIC:
            raise ValueError(
                "Magic header 'NAXX' not found at the start of the LSB stream. "
                "No valid payload present or layer too small."
            )

        payload_len = struct.unpack("<I", header[len(self.MAGIC) : self.HEADER_SIZE])[0]
        total_bits = (self.HEADER_SIZE + payload_len) * 8
        if lsbs.numel() < total_bits:
            raise ValueError(
                f"Payload truncated. Expected {payload_len} bytes, "
                f"got {(lsbs.numel() // 8) - self.HEADER_SIZE}"
            )

        payload_bits = lsbs[header_bits:total_bits]
        return self._decode_bits(payload_bits)[:payload_len]


if __name__ == "__main__":
    print("=== WeightSpaceCovertChannel defensive fixture ===\n")
    print("This demo performs an in-memory roundtrip + torch.save / torch.load\n")

    layer = torch.nn.Linear(32, 32)
    state_dict = layer.state_dict()
    layer_name = "weight"
    channel = WeightSpaceCovertChannel()
    payload = b"inert red-team telemetry marker for defensive validation"

    print(f"Original payload length: {len(payload)} bytes")
    channel.inject(state_dict, layer_name, payload)
    extracted_mem = channel.extract(state_dict, layer_name)
    print(f"In-memory extraction matches: {extracted_mem == payload}")

    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "test_model.pt"
        torch.save({"model": layer.state_dict()}, model_path)
        loaded = torch.load(model_path, weights_only=True)
        extracted_disk = channel.extract(loaded["model"], layer_name)
        print(f"After torch.save/load extraction matches: {extracted_disk == payload}")

    if extracted_mem == payload and extracted_disk == payload:
        print("\nSUCCESS: Payload survived both defensive fixture roundtrips.")
    else:
        print("\nFAILURE: Payload mismatch after extraction.")
