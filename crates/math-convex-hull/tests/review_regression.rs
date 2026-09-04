//! Review regression tests: duplicate points (P0) and scale-aware epsilon (P1).

use math_convex_hull::{ConvexHull3D, Vertex};

fn unit_cube() -> Vec<Vertex> {
    vec![
        Vertex::new(0.0, 0.0, 0.0),
        Vertex::new(1.0, 0.0, 0.0),
        Vertex::new(1.0, 1.0, 0.0),
        Vertex::new(0.0, 1.0, 0.0),
        Vertex::new(0.0, 0.0, 1.0),
        Vertex::new(1.0, 0.0, 1.0),
        Vertex::new(1.0, 1.0, 1.0),
        Vertex::new(0.0, 1.0, 1.0),
    ]
}

/// P0 regression: duplicated input points must not corrupt the main loop.
///
/// Outside-point indices address the post-dedup `unique_vertices`, so the
/// furthest-point lookup must use that slice. Duplicating every point
/// exercises the dedup/index path and must yield the same hull.
#[test]
fn duplicate_points_yield_same_hull() {
    let base = unit_cube();
    let hull_base = ConvexHull3D::build(&base).expect("base cube hull must build");

    // Each point repeated 3x, plus shuffled-ish order (reversed).
    let mut duplicated: Vec<Vertex> = Vec::with_capacity(base.len() * 3);
    for p in base.iter().rev() {
        duplicated.push(*p);
        duplicated.push(*p);
        duplicated.push(*p);
    }

    let hull_dup = ConvexHull3D::build(&duplicated).expect("duplicated cube hull must build");

    assert_eq!(
        hull_base.num_faces(),
        hull_dup.num_faces(),
        "duplicate points changed face count"
    );
    assert_eq!(hull_base.num_vertices(), 8);
    assert_eq!(hull_dup.num_vertices(), 8);

    let rel_vol = (hull_base.volume() - hull_dup.volume()).abs() / hull_base.volume();
    assert!(
        rel_vol < 1e-9,
        "duplicate points changed volume: {} vs {}",
        hull_base.volume(),
        hull_dup.volume()
    );

    // Tetrahedron with duplicates: minimal hull still exact.
    let tet = vec![
        Vertex::new(0.0, 0.0, 0.0),
        Vertex::new(1.0, 0.0, 0.0),
        Vertex::new(0.0, 1.0, 0.0),
        Vertex::new(0.0, 0.0, 1.0),
        Vertex::new(0.0, 0.0, 0.0),
        Vertex::new(1.0, 0.0, 0.0),
    ];
    let hull_tet = ConvexHull3D::build(&tet).expect("duplicated tet hull must build");
    assert_eq!(hull_tet.num_faces(), 4);
    assert!((hull_tet.volume() - 1.0 / 6.0).abs() < 1e-9);
}

/// P1 property: hull is invariant under translation and uniform large scaling.
///
/// A scale-aware relative tolerance must classify visibility the same way
/// for a unit cube, the cube translated far from the origin, and the cube
/// scaled up by 1e6.
#[test]
fn translated_and_large_scale_hulls_match() {
    let base = unit_cube();
    let hull_base = ConvexHull3D::build(&base).expect("base cube hull must build");

    // Translation far from origin.
    let offset = Vertex::new(1.0e6, -2.0e6, 5.0e5);
    let translated: Vec<Vertex> = base.iter().map(|v| v.add(&offset)).collect();
    let hull_t = ConvexHull3D::build(&translated).expect("translated hull must build");

    assert_eq!(
        hull_base.num_faces(),
        hull_t.num_faces(),
        "translation changed face count: {} vs {}",
        hull_base.num_faces(),
        hull_t.num_faces()
    );
    let rel_vol = (hull_base.volume() - hull_t.volume()).abs() / hull_base.volume();
    assert!(
        rel_vol < 1e-6,
        "translation changed volume: {} vs {}",
        hull_base.volume(),
        hull_t.volume()
    );

    // Uniform large scale: volume ~ s^3, area ~ s^2.
    let s = 1.0e6;
    let scaled: Vec<Vertex> = base.iter().map(|v| v.scale(s)).collect();
    let hull_s = ConvexHull3D::build(&scaled).expect("large-scale hull must build");
    assert_eq!(
        hull_base.num_faces(),
        hull_s.num_faces(),
        "large scale changed face count: {} vs {}",
        hull_base.num_faces(),
        hull_s.num_faces()
    );
    let expected_vol = hull_base.volume() * s * s * s;
    let rel_vol_s = (hull_s.volume() - expected_vol).abs() / expected_vol;
    assert!(
        rel_vol_s < 1e-6,
        "large-scale volume wrong: {} vs {}",
        hull_s.volume(),
        expected_vol
    );
    let expected_sa = hull_base.surface_area() * s * s;
    let rel_sa = (hull_s.surface_area() - expected_sa).abs() / expected_sa;
    assert!(
        rel_sa < 1e-6,
        "large-scale surface area wrong: {} vs {}",
        hull_s.surface_area(),
        expected_sa
    );
}
