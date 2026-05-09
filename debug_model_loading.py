#!/usr/bin/env python3
"""
Debug script to test model loading without training.

This will show exactly which model component is hanging during initialization.
"""

import argparse
import sys
import torch


def main():
    parser = argparse.ArgumentParser(description="Debug model loading")
    parser.add_argument(
        "--pretrained",
        default="stabilityai/stable-diffusion-xl-base-1.0",
        help="Pretrained model ID or path",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        default=False,
        help="Only load local files (no downloads)",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to load on",
    )
    parser.add_argument(
        "--num-controlnets",
        type=int,
        default=3,
        help="Number of ControlNets to create (1-3, start with 1 to test)",
    )
    args = parser.parse_args()

    if args.num_controlnets < 1 or args.num_controlnets > 3:
        print("ERROR: num_controlnets must be 1-3")
        sys.exit(1)

    print("=" * 80)
    print("MODEL LOADING DEBUG SCRIPT")
    print("=" * 80)
    print()

    # Check CUDA
    print("[STEP 0] Checking device...")
    print(f"  Device requested: {args.device}")
    if args.device == "cuda":
        print(f"  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  CUDA device: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print()

    # Import
    print("[STEP 1] Importing modules...")
    try:
        from my_model import SDXL15MultiControlNetBEV
        print("  ✓ Successfully imported SDXL15MultiControlNetBEV")
    except Exception as e:
        print(f"  ✗ Failed to import: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    print()

    # Create model instance (doesn't load weights yet)
    print("[STEP 2] Creating model instance...")
    try:
        model = SDXL15MultiControlNetBEV(
            pretrained=args.pretrained,
            prompt="a high quality image",
            local_files_only=args.local_files_only,
            freeze_unet=True,
        )
        print("  ✓ Model instance created (weights not loaded yet)")
    except Exception as e:
        print(f"  ✗ Failed to create instance: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    print()

    # Initialize model (loads weights)
    print(f"[STEP 3] Initializing model (loading weights - THIS MIGHT TAKE A WHILE)...")
    print(f"  Testing with {args.num_controlnets} ControlNet(s)")
    print("  If this hangs at ControlNet creation, you can:")
    print("    - Wait 5-10 minutes (ControlNet creation is slow)")
    print("    - Press Ctrl+C to cancel and try with fewer ControlNets")
    print()
    try:
        device = torch.device(args.device)
        print(f"  Loading {args.num_controlnets} ControlNet(s)...", flush=True)
        model.initialize(device, n_controlnets=args.num_controlnets)
        print("  ✓ Model initialized successfully")
    except KeyboardInterrupt:
        print("\n  ✗ Interrupted by user (Ctrl+C)")
        print("\n  SOLUTION: ControlNet creation is slow. Try with fewer ControlNets:")
        print(f"    python debug_model_loading.py --num-controlnets 1")
        sys.exit(1)
    except Exception as e:
        print(f"  ✗ Failed to initialize: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    print()

    # Check model state
    print("[STEP 4] Checking model state...")
    print(f"  Model device: {next(model.parameters()).device}")
    print(f"  Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.1f}M")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    print()

    # Summary
    print("=" * 80)
    print("✓ ALL CHECKS PASSED")
    print("=" * 80)
    print()
    print("Model loading works correctly!")
    print()


if __name__ == "__main__":
    main()
