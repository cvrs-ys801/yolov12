#!/usr/bin/env python3
"""
YOLOv12-DMMA 模型验证脚本
验证新创建的模型配置是否可正常加载和运行

作者: Antigravity
日期: 2025-12-29
"""

import sys
import os
import argparse

# Add project root to path
sys.path.append(os.getcwd())

import torch


def verify_model(cfg_path: str, verbose: bool = True):
    """验证模型配置是否可正常加载和运行前向传播。
    
    Args:
        cfg_path: 模型配置文件路径
        verbose: 是否打印详细信息
    
    Returns:
        bool: 验证是否成功
    """
    from ultralytics.nn.tasks import DetectionModel
    from ultralytics.utils import yaml_load
    
    if verbose:
        print(f"Loading config from {cfg_path}...")
    
    try:
        cfg = yaml_load(cfg_path)
        # Use a compatible nc (number of classes)
        cfg['nc'] = 1 
        
        if verbose:
            print("Instantiating model...")
        model = DetectionModel(cfg=cfg, ch=3, nc=1)
        
        if verbose:
            print("Model instantiated successfully.")
        
        # Create dummy input
        img_size = 640
        dummy_input = torch.randn(1, 3, img_size, img_size)
        
        if verbose:
            print(f"Running forward pass with input shape {dummy_input.shape}...")
        
        # Set to eval mode for inference
        model.eval()
        with torch.no_grad():
            results = model(dummy_input)
        
        if verbose:
            print("Forward pass successful.")
            print("Output shapes:")
            if isinstance(results, (list, tuple)):
                for i, res in enumerate(results):
                    if isinstance(res, torch.Tensor):
                        print(f"  Result {i}: {res.shape}")
                    else:
                        print(f"  Result {i}: {type(res)}")
            else:
                print(f"  Result: {results.shape}")

        # Check for module types
        if verbose:
            print("\nChecking for DMMA/ECA/SPPF modules...")
        
        dmma_count = 0
        eca_count = 0
        spatial_count = 0
        sppf_count = 0
        msdmma_count = 0
        
        for name, m in model.named_modules():
            class_name = m.__class__.__name__
            if 'C2fDMMA' in class_name:
                dmma_count += 1
            if 'DMMAChannelAttention' in class_name:
                eca_count += 1
            if 'DMMAChannelSpatialAttention' in class_name:
                spatial_count += 1
            if 'SPPF' in class_name:
                sppf_count += 1
            if 'MSDMMALayer' in class_name:
                msdmma_count += 1
                
        if verbose:
            print(f"Found {dmma_count} C2fDMMA modules.")
            print(f"Found {eca_count} DMMAChannelAttention (ECA) modules.")
            print(f"Found {spatial_count} DMMAChannelSpatialAttention modules.")
            print(f"Found {sppf_count} SPPF modules.")
            print(f"Found {msdmma_count} MSDMMALayer (multi-scale) modules.")
            
            if dmma_count > 0:
                print("\n✓ SUCCESS: DMMA modules are present.")
            if sppf_count > 0:
                print("✓ SUCCESS: SPPF module is present.")
            if msdmma_count > 0:
                print("✓ SUCCESS: Multi-scale DMMA is enabled.")
            
            # Get model info
            print("\n" + "=" * 50)
            print("Model Summary:")
            print("=" * 50)
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"Total parameters: {total_params:,}")
            print(f"Trainable parameters: {trainable_params:,}")
            print(f"Model size: {total_params * 4 / 1024 / 1024:.2f} MB (FP32)")
        
        return True
        
    except Exception as e:
        print(f"\n✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def verify_all_models():
    """验证所有 DMMA 模型配置。"""
    models = [
        "ultralytics/cfg/models/v12/yolov12-dmma-p2-efficient.yaml",
        "ultralytics/cfg/models/v12/yolov12-dmma-p2-advanced.yaml",
        "ultralytics/cfg/models/v12/yolov12-dmma-p2-dasa.yaml",
        "ultralytics/cfg/models/v12/yolov12-dmma-p2-ultimate.yaml",
        "ultralytics/cfg/models/v12/yolov12-dmma-ms.yaml",
        "ultralytics/cfg/models/v12/yolov12-dmma-p2.yaml",
    ]
    
    print("=" * 60)
    print("Verifying all DMMA model configurations")
    print("=" * 60)
    
    results = {}
    for model_path in models:
        if os.path.exists(model_path):
            print(f"\n--- Testing: {os.path.basename(model_path)} ---")
            success = verify_model(model_path, verbose=False)
            results[model_path] = success
            status = "✓ PASS" if success else "✗ FAIL"
            print(f"Result: {status}")
        else:
            print(f"\n--- Skipping (not found): {model_path} ---")
            results[model_path] = None
    
    print("\n" + "=" * 60)
    print("Summary:")
    print("=" * 60)
    for path, success in results.items():
        name = os.path.basename(path)
        if success is None:
            status = "SKIPPED"
        elif success:
            status = "✓ PASS"
        else:
            status = "✗ FAIL"
        print(f"  {name}: {status}")
    
    return all(v for v in results.values() if v is not None)


def test_dmma_module():
    """直接测试 C2fDMMA 模块（不依赖完整模型）。"""
    print("=" * 60)
    print("Testing C2fDMMA module directly")
    print("=" * 60)
    
    try:
        from ultralytics.nn.modules.block import C2fDMMA
        
        # Test single-scale DMMA
        print("\n1. Single-scale DMMA (window_size=8)...")
        m1 = C2fDMMA(256, 256, n=2, window_size=8, num_heads=8, shift=True, mlp_ratio=2.0)
        x1 = torch.randn(1, 256, 40, 40)
        y1 = m1(x1)
        print(f"   Input: {x1.shape} -> Output: {y1.shape}")
        print("   ✓ Single-scale DMMA works!")
        
        # Test multi-scale DMMA with density gate
        print("\n2. Multi-scale DMMA with density gate (window_size=[4, 8])...")
        m2 = C2fDMMA(256, 256, n=2, window_size=[4, 8], num_heads=8, shift=True, mlp_ratio=2.0)
        x2 = torch.randn(1, 256, 40, 40)
        y2 = m2(x2)
        print(f"   Input: {x2.shape} -> Output: {y2.shape}")
        print("   ✓ Multi-scale DMMA with density gate works!")
        
        return True
        
    except Exception as e:
        print(f"\n✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dasa_module():
    """测试 DASA 和 SAQK-Mask 模块。"""
    print("=" * 60)
    print("Testing DASA and SAQK-Mask modules")
    print("=" * 60)
    
    try:
        from ultralytics.nn.modules.transformer import DensityAwareGate, SAQKMask, DASALayer
        from ultralytics.nn.modules.block import C2fDASA, C2fDMMA_SAQK
        
        # Test DensityAwareGate
        print("\n1. DensityAwareGate...")
        gate = DensityAwareGate(256, num_branches=3)
        x = torch.randn(2, 256, 32, 32)
        weights, density_map = gate(x)
        print(f"   Input: {x.shape}")
        print(f"   Weights: {weights.shape}, Density map: {density_map.shape}")
        print(f"   Weights sum to 1: {weights.sum(dim=1)}")
        print("   ✓ DensityAwareGate works!")
        
        # Test SAQKMask
        print("\n2. SAQKMask...")
        saqk = SAQKMask(256, scale_factor=0.5)
        mask, fg_prob = saqk(x)
        print(f"   Input: {x.shape}")
        print(f"   Mask: {mask.shape}, range [{mask.min():.2f}, {mask.max():.2f}]")
        print(f"   FG prob: {fg_prob.shape}")
        print("   ✓ SAQKMask works!")
        
        # Test DASALayer
        print("\n3. DASALayer...")
        dasa = DASALayer(256, num_heads=4, window_sizes=(4, 8, 16))
        y = dasa(x)
        print(f"   Input: {x.shape} -> Output: {y.shape}")
        print("   ✓ DASALayer works!")
        
        # Test C2fDASA
        print("\n4. C2fDASA...")
        c2f_dasa = C2fDASA(256, 256, n=1, window_sizes=(4, 8, 16), num_heads=4)
        y = c2f_dasa(x)
        print(f"   Input: {x.shape} -> Output: {y.shape}")
        print("   ✓ C2fDASA works!")
        
        # Test C2fDMMA_SAQK
        print("\n5. C2fDMMA_SAQK...")
        c2f_saqk = C2fDMMA_SAQK(256, 256, n=1, window_size=[4, 8], num_heads=8, scale_factor=0.5)
        y = c2f_saqk(x)
        print(f"   Input: {x.shape} -> Output: {y.shape}")
        print("   ✓ C2fDMMA_SAQK works!")
        
        print("\n" + "=" * 60)
        print("All DASA/SAQK modules passed!")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\n✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_spatial_attention():

    """测试新增的空间注意力模块。"""
    print("=" * 60)
    print("Testing DMMAChannelSpatialAttention module")
    print("=" * 60)
    
    try:
        from ultralytics.nn.modules.transformer import DMMAChannelSpatialAttention
        
        print("\nCreating DMMAChannelSpatialAttention(256)...")
        attn = DMMAChannelSpatialAttention(256, spatial_kernel=7)
        
        x = torch.randn(2, 256, 32, 32)
        print(f"Input shape: {x.shape}")
        
        y = attn(x)
        print(f"Output shape: {y.shape}")
        
        assert x.shape == y.shape, "Input and output shapes should match!"
        print("\n✓ DMMAChannelSpatialAttention works correctly!")
        
        return True
        
    except Exception as e:
        print(f"\n✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv12-DMMA Model Verification")
    parser.add_argument("--model", type=str, 
                        default="ultralytics/cfg/models/v12/yolov12-dmma-p2-ultimate.yaml",
                        help="Path to model config YAML")
    parser.add_argument("--all", action="store_true", 
                        help="Verify all DMMA model configurations")
    parser.add_argument("--module", action="store_true",
                        help="Test C2fDMMA module only")
    parser.add_argument("--spatial", action="store_true",
                        help="Test DMMAChannelSpatialAttention module")
    parser.add_argument("--dasa", action="store_true",
                        help="Test DASA and SAQK-Mask modules")
    
    args = parser.parse_args()
    
    if args.all:
        verify_all_models()
    elif args.module:
        test_dmma_module()
    elif args.spatial:
        test_spatial_attention()
    elif args.dasa:
        test_dasa_module()
    else:
        verify_model(args.model)

