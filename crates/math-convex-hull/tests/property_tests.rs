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

        // Faces may use different vertex indices after deduplication, but the
        // geometric faces (as unordered sets of vertex coordinates) must match.
        let mut faces_a: Vec<[Vertex; 3]> = hull_a
            .faces()
            .iter()
            .map(|f| {
                [
                    hull_a.vertices()[f.v0],
                    hull_a.vertices()[f.v1],
                    hull_a.vertices()[f.v2],
                ]
            })
            .collect();
        let mut faces_b: Vec<[Vertex; 3]> = hull_b
            .faces()
            .iter()
            .map(|f| {
                [
                    hull_b.vertices()[f.v0],
                    hull_b.vertices()[f.v1],
                    hull_b.vertices()[f.v2],
                ]
            })
            .collect();

        for face in &mut faces_a {
            face.sort_by(|a, b| {
                a.x.partial_cmp(&b.x)
                    .unwrap()
                    .then(a.y.partial_cmp(&b.y).unwrap())
                    .then(a.z.partial_cmp(&b.z).unwrap())
            });
        }
        for face in &mut faces_b {
            face.sort_by(|a, b| {
                a.x.partial_cmp(&b.x)
                    .unwrap()
                    .then(a.y.partial_cmp(&b.y).unwrap())
                    .then(a.z.partial_cmp(&b.z).unwrap())
            });
        }
        faces_a.sort_by(|a, b| {
            a[0].x.partial_cmp(&b[0].x)
                .unwrap()
                .then(a[0].y.partial_cmp(&b[0].y).unwrap())
                .then(a[0].z.partial_cmp(&b[0].z).unwrap())
        });
        faces_b.sort_by(|a, b| {
            a[0].x.partial_cmp(&b[0].x)
                .unwrap()
                .then(a[0].y.partial_cmp(&b[0].y).unwrap())
                .then(a[0].z.partial_cmp(&b[0].z).unwrap())
        });

        prop_assert_eq!(faces_a.len(), faces_b.len(), "face count changed");
        for (fa, fb) in faces_a.iter().zip(faces_b.iter()) {
            for i in 0..3 {
                prop_assert!(
                    approx_eq(fa[i].x, fb[i].x, eps)
                        && approx_eq(fa[i].y, fb[i].y, eps)
                        && approx_eq(fa[i].z, fb[i].z, eps),
                    "geometric face changed after adding duplicates: {:?} vs {:?}",
                    fa,
                    fb
                );
            }
        }
    }
}
