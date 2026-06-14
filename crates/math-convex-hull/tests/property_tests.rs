// ============================================================================
// Property-Based Tests for math-convex-hull
// ============================================================================
//
// Invariants checked:
//   - All input points lie inside or on the 3D convex hull.
//   - Hull vertices are a subset of the input points.
//   - Duplicating input points does not change the hull.

use math_convex_hull::{ConvexHull3D, Vertex};
use proptest::prelude::*;

fn vertex_strategy() -> impl Strategy<Value = Vertex> {
    (-10.0f64..10.0f64, -10.0f64..10.0f64, -10.0f64..10.0f64)
        .prop_map(|(x, y, z)| Vertex::new(x, y, z))
}

fn approx_eq(a: f64, b: f64, eps: f64) -> bool {
    (a - b).abs() <= eps
}

/// True if `point` lies inside `hull` or on its boundary.
fn point_in_hull(hull: &ConvexHull3D, point: &Vertex, eps: f64) -> bool {
    let vertices = hull.vertices();
    let n = vertices.len();
    if n == 0 {
        return false;
    }

    // Centroid of hull vertices is guaranteed to be inside the hull.
    let centroid = vertices
        .iter()
        .fold(Vertex::new(0.0, 0.0, 0.0), |a, b| a.add(b))
        .scale(1.0 / n as f64);

    for face in hull.faces() {
        let normal = face.normal(vertices);
        let v0 = vertices[face.v0];
        let signed = normal.dot(&point.sub(&v0));
        let centroid_signed = normal.dot(&centroid.sub(&v0));

        if centroid_signed > eps {
            // Normal points inward: interior is the positive side.
            if signed < -eps {
                return false;
            }
        } else if centroid_signed < -eps {
            // Normal points outward: interior is the negative side.
            if signed > eps {
                return false;
            }
        }
    }
    true
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(32))]

    /// INVARIANT: every input point lies inside the hull or on its boundary.
    #[test]
    fn all_input_points_inside_or_on_hull(points in prop::collection::vec(vertex_strategy(), 4..30)) {
        let hull = match ConvexHull3D::build(&points) {
            Ok(h) => h,
            Err(_) => return Ok(()), // skip degenerate inputs
        };

        let eps = 1e-6;
        for (i, p) in points.iter().enumerate() {
            prop_assert!(
                point_in_hull(&hull, p, eps),
                "input point {} {:?} is outside the convex hull",
                i,
                p
            );
        }
    }

    /// INVARIANT: hull vertices are a subset of the input points.
    #[test]
    fn hull_vertices_are_subset_of_input(points in prop::collection::vec(vertex_strategy(), 4..30)) {
        let hull = match ConvexHull3D::build(&points) {
            Ok(h) => h,
            Err(_) => return Ok(()),
        };

        let eps = 1e-9;
        for (i, hv) in hull.vertices().iter().enumerate() {
            prop_assert!(
                points.iter().any(|p| approx_eq(p.x, hv.x, eps)
                    && approx_eq(p.y, hv.y, eps)
                    && approx_eq(p.z, hv.z, eps)),
                "hull vertex {} {:?} is not present in input",
                i,
                hv
            );
        }
    }

    /// INVARIANT: duplicating input points does not change the hull.
    #[test]
    fn duplicate_points_do_not_change_hull(points in prop::collection::vec(vertex_strategy(), 4..20)) {
        let hull_a = match ConvexHull3D::build(&points) {
            Ok(h) => h,
            Err(_) => return Ok(()),
        };

        let mut duplicated = points.clone();
        for p in &points {
            duplicated.push(*p);
        }

        let hull_b = match ConvexHull3D::build(&duplicated) {
            Ok(h) => h,
            Err(_) => return Ok(()),
        };

        prop_assert_eq!(
            hull_a.num_faces(),
            hull_b.num_faces(),
            "duplicate points changed the number of hull faces"
        );
        prop_assert_eq!(
            hull_a.num_vertices(),
            hull_b.num_vertices(),
            "duplicate points changed the number of hull vertices"
        );

        let eps = 1e-9;
        for (va, vb) in hull_a.vertices().iter().zip(hull_b.vertices().iter()) {
            prop_assert!(
                approx_eq(va.x, vb.x, eps)
                    && approx_eq(va.y, vb.y, eps)
                    && approx_eq(va.z, vb.z, eps),
                "hull vertex changed after adding duplicates: {:?} vs {:?}",
                va,
                vb
            );
        }

        // The triangulation of non-simplicial faces may differ, so we compare
        // the geometric hull: vertex set, volume and surface area.
        prop_assert!(
            approx_eq(hull_a.volume(), hull_b.volume(), 1e-9),
            "duplicate points changed hull volume: {} vs {}",
            hull_a.volume(),
            hull_b.volume()
        );
        prop_assert!(
            approx_eq(hull_a.surface_area(), hull_b.surface_area(), 1e-9),
            "duplicate points changed hull surface area: {} vs {}",
            hull_a.surface_area(),
            hull_b.surface_area()
        );

        let mut verts_a: Vec<_> = hull_a.vertices().to_vec();
        let mut verts_b: Vec<_> = hull_b.vertices().to_vec();
        verts_a.sort_by(|a, b| {
            a.x.partial_cmp(&b.x)
                .unwrap()
                .then(a.y.partial_cmp(&b.y).unwrap())
                .then(a.z.partial_cmp(&b.z).unwrap())
        });
        verts_b.sort_by(|a, b| {
            a.x.partial_cmp(&b.x)
                .unwrap()
                .then(a.y.partial_cmp(&b.y).unwrap())
                .then(a.z.partial_cmp(&b.z).unwrap())
        });
        prop_assert_eq!(verts_a.len(), verts_b.len(), "vertex count changed");
        for (va, vb) in verts_a.iter().zip(verts_b.iter()) {
            prop_assert!(
                approx_eq(va.x, vb.x, eps)
                    && approx_eq(va.y, vb.y, eps)
                    && approx_eq(va.z, vb.z, eps),
                "hull vertex set changed after adding duplicates: {:?} vs {:?}",
                va,
                vb
            );
        }
    }
}
