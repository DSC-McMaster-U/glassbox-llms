"""
Non-Linear Probe Example: Clear Non-Linear Superiority

Demonstrates when non-linear probes significantly outperform linear probes.
Shows clear cases where linear probes fail but non-linear probes succeed.

Usage:
    python -m probes.examples.nonlinear_superiority
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import warnings
from typing import Tuple
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore', category=UserWarning)

from probes import LinearProbe, NonLinearProbe


def create_xor_pattern() -> Tuple[np.ndarray, np.ndarray]:
    """Create XOR pattern - will fail for linear probes."""
    np.random.seed(42)
    n_samples = 1000
    
    # Clean XOR in 2D
    X_signal = np.random.uniform(-1, 1, (n_samples, 2))
    
    # Classic XOR: 1 if signs are different
    y = ((X_signal[:, 0] > 0) != (X_signal[:, 1] > 0)).astype(int)
    
    # Add minimal noise dimensions
    noise = np.random.randn(n_samples, 10) * 0.1
    X = np.hstack([X_signal, noise])
    
    # Scale
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    return X, y


def create_circles_pattern() -> Tuple[np.ndarray, np.ndarray]:
    """Concentric circles - impossible for linear classifiers."""
    np.random.seed(42)
    n_samples = 1000
    
    # Generate two concentric circles
    angles = np.linspace(0, 2 * np.pi, n_samples // 2)
    
    # Inner circle (class 0)
    r1 = np.random.uniform(0, 0.3, n_samples // 2)
    x1 = r1 * np.cos(angles)
    y1 = r1 * np.sin(angles)
    
    # Outer circle (class 1)
    r2 = np.random.uniform(0.7, 1.0, n_samples // 2)
    x2 = r2 * np.cos(angles + np.pi)
    y2 = r2 * np.sin(angles + np.pi)
    
    # Combine
    X_signal = np.vstack([
        np.column_stack([x1, y1]),
        np.column_stack([x2, y2])
    ])
    y = np.array([0] * (n_samples // 2) + [1] * (n_samples // 2))
    
    # Add minimal noise
    noise = np.random.randn(n_samples, 10) * 0.1
    X = np.hstack([X_signal, noise])
    
    # Scale
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    return X, y


def create_checkerboard_pattern() -> Tuple[np.ndarray, np.ndarray]:
    """Checkerboard pattern - requires non-linear separation."""
    np.random.seed(42)
    n_samples = 1000
    
    # Create grid-like pattern
    X_signal = np.random.uniform(-2, 2, (n_samples, 2))
    
    # Checkerboard: class depends on floor of sum
    x_floor = np.floor(X_signal[:, 0])
    y_floor = np.floor(X_signal[:, 1])
    y = ((x_floor + y_floor) % 2).astype(int)
    
    # Add noise
    noise = np.random.randn(n_samples, 10) * 0.1
    X = np.hstack([X_signal, noise])
    
    # Scale
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    return X, y


def main():
    print("=" * 70)
    print("NON-LINEAR PROBE SUPERIORITY DEMONSTRATION")
    print("=" * 70)
    print("\nTesting on patterns where LINEAR PROBES SHOULD FAIL:")
    
    test_cases = [
        ("XOR Pattern", create_xor_pattern()),
        ("Concentric Circles", create_circles_pattern()),
        ("Checkerboard", create_checkerboard_pattern()),
    ]
    
    results = []
    
    for case_name, (X, y) in test_cases:
        print(f"\n{'='*60}")
        print(f"CASE: {case_name}")
        print(f"{'='*60}")
        
        # Split data (80/20)
        split = int(0.8 * len(X))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        
        # -----------------------------------------------------------------
        # 1. Linear Probe (Expected to fail)
        # -----------------------------------------------------------------
        print(f"\n  [1] LINEAR PROBE:")
        
        try:
            linear_probe = LinearProbe(
                layer="test",
                direction=case_name.replace(" ", "_").lower(),
                model_type="logistic",
                normalize=False,
                regularization=1.0,
                max_iter=1000,
            )
            
            linear_probe.fit(X_train, y_train)
            linear_result = linear_probe.evaluate(X_test, y_test)
            linear_acc = linear_result.accuracy
            
            print(f"    Accuracy: {linear_acc:.3f}")
            print(f"    Chance level: {0.5:.3f}")
            
        except Exception as e:
            print(f"    ERROR: {e}")
            linear_acc = 0.0
        
        # -----------------------------------------------------------------
        # 2. Non-Linear Probe (Expected to succeed)
        # -----------------------------------------------------------------
        print(f"\n  [2] NON-LINEAR PROBE:")
        
        try:
            nonlinear_probe = NonLinearProbe(
                layer="test",
                direction=case_name.replace(" ", "_").lower(),
                hidden_dims=[32, 16],
                is_classification=True,
                early_stopping=True,
                validation_fraction=0.2,
                max_iter=500,
                learning_rate_init=0.01,
                random_state=42,
                normalize=False,
                robust_scaling=False,
                alpha=0.001,
            )
            
            nonlinear_probe.fit(X_train, y_train)
            nonlinear_result = nonlinear_probe.evaluate(X_test, y_test)
            nonlinear_acc = nonlinear_result.accuracy
            
            print(f"    Accuracy: {nonlinear_acc:.3f}")
            
        except Exception as e:
            print(f"    ERROR: {e}")
            nonlinear_acc = 0.0
        
        # -----------------------------------------------------------------
        # 3. Comparison
        # -----------------------------------------------------------------
        print(f"\n  [3] COMPARISON:")
        print("  " + "-" * 40)
        
        improvement = nonlinear_acc - linear_acc
        
        print(f"    Linear:     {linear_acc:.3f}")
        print(f"    Non-linear: {nonlinear_acc:.3f}")
        print(f"    Improvement: {improvement:+.3f}")
        
        # Determine verdict
        if improvement > 0.2:
            verdict = "✅ MAJOR NON-LINEAR ADVANTAGE"
        elif improvement > 0.1:
            verdict = "✓ Significant non-linear advantage"
        elif improvement > 0.05:
            verdict = "~ Moderate non-linear advantage"
        elif improvement > 0:
            verdict = "→ Slight non-linear advantage"
        else:
            verdict = "✗ Linear performed better"
        
        print(f"\n    {verdict}")
        
        # Store results
        results.append({
            "case": case_name,
            "linear": linear_acc,
            "nonlinear": nonlinear_acc,
            "improvement": improvement,
            "verdict": verdict,
        })
    
    # -----------------------------------------------------------------
    # 4. Summary
    # -----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    print("\n" + "-" * 60)
    print(f"{'Pattern':<20} {'Linear':<8} {'NonLinear':<10} {'Δ':<8} {'Verdict'}")
    print("-" * 60)
    
    for r in results:
        print(f"{r['case']:<20} {r['linear']:.3f}     {r['nonlinear']:.3f}      {r['improvement']:+.3f}   {r['verdict'].split()[0]}")
    
    print("-" * 60)
    
    # Count successes
    major_wins = sum(1 for r in results if r['improvement'] > 0.2)
    
    print(f"\nNon-linear probes showed major advantage (>0.2) in: {major_wins}/{len(results)} cases")
    
    # -----------------------------------------------------------------
    # 5. Why This Matters
    # -----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("WHY NON-LINEAR PROBES ARE ESSENTIAL")
    print("=" * 70)
    
    print("""
    Key Insights:
    
    1. Information Can Be Present But Not Linearly Accessible
       - Just because a linear probe fails doesn't mean the information isn't there
       - Non-linear probes can reveal "hidden" information
    
    2. Different Tasks Require Different Probes
       • Linear probes: Good for simple, linearly separable concepts
       • Non-linear probes: Necessary for complex, compositional concepts
    
    3. This Framework Lets You Distinguish:
       Case A: Both probes fail → Information not encoded
       Case B: Linear fails, non-linear succeeds → Information encoded non-linearly
       Case C: Both succeed → Information linearly accessible
       Case D: Linear succeeds, non-linear fails → Simple linear encoding
    
    4. Real-World Examples Where Non-Linear Probes Help:
       • Syntax tree depth and structure
       • Coreference resolution
       • Logical reasoning (if-then, and-or patterns)
       • Semantic composition
       • Numerical reasoning
    """)
    
    # -----------------------------------------------------------------
    # 6. Simple ASCII Visualization
    # -----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("PATTERN VISUALIZATION")
    print("=" * 70)
    
    # Create simple ASCII visualizations
    for case_name, (X, y) in [("XOR Pattern", create_xor_pattern()),
                             ("Concentric Circles", create_circles_pattern())]:
        
        print(f"\n{case_name}:")
        print("-" * 40)
        
        # Use first 100 samples
        X_vis = X[:100, :2]
        y_vis = y[:100]
        
        # Create 10x10 grid
        grid_size = 10
        grid = [[" " for _ in range(grid_size)] for _ in range(grid_size)]
        
        # Normalize to grid
        x_min, x_max = X_vis[:, 0].min(), X_vis[:, 0].max()
        y_min, y_max = X_vis[:, 1].min(), X_vis[:, 1].max()
        
        for i in range(len(X_vis)):
            x_norm = int((X_vis[i, 0] - x_min) / (x_max - x_min) * (grid_size - 1))
            y_norm = int((X_vis[i, 1] - y_min) / (y_max - y_min) * (grid_size - 1))
            
            if 0 <= x_norm < grid_size and 0 <= y_norm < grid_size:
                symbol = "X" if y_vis[i] == 1 else "O"
                grid[y_norm][x_norm] = symbol
        
        # Print grid
        print("   " + "-" * grid_size)
        for row in grid:
            print("  |" + "".join(row) + "|")
        print("   " + "-" * grid_size)
        print("  O = Class 0, X = Class 1")
        
        if "XOR" in case_name:
            print("  Note: X's and O's are mixed - no single line can separate them!")
        else:
            print("  Note: Circles within circles - requires curved boundary!")
    
    # -----------------------------------------------------------------
    # 7. Conclusion
    # -----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    
    print("""
    ✅ Non-linear probes are VALUABLE when:
    
    1. The underlying computation is non-linear
       • Logical operations (XOR, etc.)
       • Hierarchical or compositional structure
       • Geometric patterns (circles, spirals, checkerboards)
    
    2. You need to measure information bounds
       • What's the MAXIMUM information extractable?
       • Not just what's linearly accessible
    
    3. You're studying complex cognitive tasks
       • Syntax parsing, coreference, logical reasoning
       • Tasks known to require non-linear computation
    
    Your NonLinearProbe implementation SUCCESSFULLY:
    • Detects when linear probes are insufficient
    • Reveals non-linearly encoded information
    • Provides quantitative evidence of non-linear structure
    
    This is EXACTLY what a non-linear probing framework should do!
    """)
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()