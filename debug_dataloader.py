#!/usr/bin/env python3
"""
Debug script to test data loader without multi-worker processes.

If the script hangs, it indicates an issue with:
1. The bev_outputs directory structure
2. The annotations.json file
3. The OCC3D base directory
4. The RGB paths not matching stems

Run this to identify data loading issues before training.
"""

import argparse
import sys
import os


def main():
    parser = argparse.ArgumentParser(description="Debug data loader")
    parser.add_argument("--bev-root", default="./bev_outputs", help="Path to BEV outputs")
    parser.add_argument("--occ3d-base-dir", default="/home/vbontal1/Downloads/Occupancy3D", help="OCC3D base dir")
    parser.add_argument("--annotations-json", default=None, help="Annotations JSON path")
    parser.add_argument("--resolution", type=int, default=128, help="Resolution")
    parser.add_argument("--load-rgb", action="store_true", default=True, help="Load RGB images")
    parser.add_argument("--no-load-rgb", dest="load_rgb", action="store_false", help="Skip RGB loading")
    parser.add_argument("--num-samples", type=int, default=5, help="Number of samples to load")
    args = parser.parse_args()

    print("=" * 80)
    print("DATALOADER DEBUG SCRIPT")
    print("=" * 80)
    print()

    # Step 1: Check paths
    print("[STEP 1] Checking paths...")
    print(f"  BEV root: {args.bev_root}")
    print(f"  BEV root exists: {os.path.isdir(args.bev_root)}")
    
    if args.load_rgb:
        print(f"  OCC3D base dir: {args.occ3d_base_dir}")
        print(f"  OCC3D base dir exists: {os.path.isdir(args.occ3d_base_dir)}")
        
        annot = args.annotations_json or os.path.join(args.occ3d_base_dir, "annotations.json")
        print(f"  Annotations JSON: {annot}")
        print(f"  Annotations JSON exists: {os.path.isfile(annot)}")
    print()

    # Step 2: Import and create dataset
    print("[STEP 2] Importing data loader modules...")
    try:
        from data_loader import BEVDataset
        print("  ✓ Successfully imported BEVDataset")
    except Exception as e:
        print(f"  ✗ Failed to import: {e}")
        sys.exit(1)
    print()

    print("[STEP 3] Creating BEVDataset (with RGB loading)...")
    try:
        ds = BEVDataset(
            root=args.bev_root,
            resolution=args.resolution,
            load_rgb=args.load_rgb,
            occ3d_base_dir=args.occ3d_base_dir,
            annotations_json=args.annotations_json,
        )
        print(f"  ✓ Dataset created successfully")
        print(f"  Number of samples: {len(ds)}")
    except FileNotFoundError as e:
        print(f"  ✗ File not found: {e}")
        sys.exit(1)
    except RuntimeError as e:
        print(f"  ✗ Runtime error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"  ✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    print()

    # Step 3: Load individual samples
    print(f"[STEP 4] Loading {min(args.num_samples, len(ds))} samples...")
    try:
        for i in range(min(args.num_samples, len(ds))):
            print(f"  Loading sample {i}...", end=" ", flush=True)
            sample = ds[i]
            print(f"✓ (stem={sample['stem']})")
            print(f"    - bev shape: {sample['bev'].shape}")
            print(f"    - depth shape: {sample['depth'].shape}")
            print(f"    - seg shape: {sample['seg'].shape}")
            if sample['rgb'] is not None:
                print(f"    - rgb shape: {sample['rgb'].shape}")
            else:
                print(f"    - rgb: None")
    except Exception as e:
        print(f"✗ Error loading sample {i}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    print()

    # Step 4: Test DataLoader with num_workers=0
    print("[STEP 5] Creating DataLoader with num_workers=0...")
    try:
        from data_loader import make_dataloader
        dl = make_dataloader(
            root=args.bev_root,
            batch_size=2,
            shuffle=False,
            num_workers=0,  # Single-threaded
            drop_last=False,
            resolution=args.resolution,
            load_rgb=args.load_rgb,
            occ3d_base_dir=args.occ3d_base_dir,
            annotations_json=args.annotations_json,
        )
        print(f"  ✓ DataLoader created with {len(dl)} batches")
    except Exception as e:
        print(f"  ✗ Failed to create DataLoader: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    print()

    # Step 5: Load one batch
    print("[STEP 6] Loading first batch from DataLoader...")
    try:
        for batch_idx, batch in enumerate(dl):
            print(f"  ✓ Batch {batch_idx} loaded:")
            print(f"    - bev shape: {batch['bev'].shape}")
            print(f"    - depth shape: {batch['depth'].shape}")
            print(f"    - seg shape: {batch['seg'].shape}")
            if batch['rgb'] is not None:
                print(f"    - rgb shape: {batch['rgb'].shape}")
            print(f"    - stems: {batch['stem']}")
            break  # Only load first batch
    except Exception as e:
        print(f"  ✗ Failed to load batch: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    print()

    # Summary
    print("=" * 80)
    print("✓ ALL CHECKS PASSED")
    print("=" * 80)
    print()
    print("Your data loader works correctly with num_workers=0!")
    print()
    print("NEXT STEPS:")
    print("1. If this script hangs anywhere, note the step where it hangs")
    print("2. The issue is likely in:")
    print("   - BEV directory structure")
    print("   - Annotations.json format or location")
    print("   - OCC3D directory or RGB paths")
    print()
    print("3. If this script succeeds, your issue is likely:")
    print("   - multi-worker deadlock (try num_workers=0 in config)")
    print("   - Or environment-specific issue")
    print()


if __name__ == "__main__":
    main()
