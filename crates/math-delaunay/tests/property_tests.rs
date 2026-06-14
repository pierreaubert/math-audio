// ============================================================================
// Property-Based Tests for math-delaunay
// ============================================================================
//
// Invariants checked:
//   - Delaunay triangulation of random 2D point sets has no overlapping
//     triangles and covers the convex hull (total triangle area equals hull
//     area).
//   - Adding a point far outside the convex hull strictly increases the
//     triangulated area.

use math_delaunay::Delaunay;
use proptest::prelude::*;

fn point_strategy() -> impl Strategy<Value = (f64, f64)> {
    (0.0f64..1.0f64, 0.0f64..1.0f64)
}

fn triangle_area(points: &[f64], triangles: &[usize], t: usize) -> f64 {
    let i = t * 3;
    let i0 = triangles[i];
    let i1 = triangles[i + 1];
    let i2 = triangles[i + 2];
    let (x0, y0) = (points[i0 * 2], points[i0 * 2 + 1]);
    let (x1, y1) = (points[i1 * 2], points[i1 * 2 + 1]);
    let (x2, y2) = (points[i2 * 2], points[i2 * 2 + 1]);
    ((x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0)).abs() * 0.5
}

fn hull_area(delaunay: &Delaunay) -> f64 {
    let hull = delaunay.hull();
    let points = delaunay.points();
    if hull.len() < 3 {
        return 0.0;
    }
    let mut a = 0.0;
    for i in 0..hull.len() {
        let i0 = hull[i];
        let i1 = hull[(i + 1) % hull.len()];
        a += points[i0 * 2] * points[i1 * 2 + 1] - points[i1 * 2] * points[i0 * 2 + 1];
    }
    a.abs() * 0.5
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(32))]

    /// INVARIANT: Delaunay triangles are non-overlapping and exactly cover the
    /// convex hull (their total area equals the hull polygon area).
    #[test]
    fn triangulation_covers_convex_hull(points in prop::collection::vec(point_strategy(), 4..16)) {
        let delaunay = Delaunay::from_points(&points);
        let triangles = delaunay.triangles();

        if triangles.len() < 3 || delaunay.hull().len() < 3 {
            return Ok(());
        }

        let total_triangle_area: f64 = (0..triangles.len() / 3)
            .map(|t| triangle_area(delaunay.points(), triangles, t))
            .sum();
        let hull_area = hull_area(&delaunay);

        let tol = 1e-9_f64.max(1e-9 * hull_area);
        prop_assert!(
            (total_triangle_area - hull_area).abs() <= tol,
            "triangle area {} does not match hull area {} (difference {})",
            total_triangle_area,
            hull_area,
            (total_triangle_area - hull_area).abs()
        );

        // Every triangle should be non-degenerate and have distinct vertices.
        for t in 0..triangles.len() / 3 {
            let area = triangle_area(delaunay.points(), triangles, t);
            prop_assert!(area > 1e-18, "degenerate triangle at index {}", t);
            let i0 = triangles[t * 3];
            let i1 = triangles[t * 3 + 1];
            let i2 = triangles[t * 3 + 2];
            prop_assert!(
                i0 != i1 && i1 != i2 && i0 != i2,
                "triangle {} has duplicate vertices", t
            );
        }
    }

    /// INVARIANT: adding a point far outside the convex hull strictly increases
    /// the triangulated area.
    #[test]
    fn adding_far_point_increases_area(points in prop::collection::vec(point_strategy(), 4..12)) {
        let delaunay = Delaunay::from_points(&points);
        let triangles = delaunay.triangles();
        if triangles.len() < 3 || delaunay.hull().len() < 3 {
            return Ok(());
        }
        let area_before: f64 = (0..triangles.len() / 3)
            .map(|t| triangle_area(delaunay.points(), triangles, t))
            .sum();

        let mut extended = points.clone();
        extended.push((10.0, 10.0));
        let delaunay2 = Delaunay::from_points(&extended);
        let triangles2 = delaunay2.triangles();
        if triangles2.len() < 3 || delaunay2.hull().len() < 3 {
            return Ok(());
        }
        let area_after: f64 = (0..triangles2.len() / 3)
            .map(|t| triangle_area(delaunay2.points(), triangles2, t))
            .sum();

        prop_assert!(
            area_after > area_before + 1e-6,
            "adding far point did not increase area: {} -> {}",
            area_before,
            area_after
        );
    }
}
