use graphrust::{bench_median_ms, Graph};

fn edges(node_count: usize) -> Vec<(usize, usize)> {
    let mut edges = Vec::with_capacity(node_count * 6);
    for node in 0..node_count {
        edges.push((node, (node + 1) % node_count));
        for step in 1..=5 {
            edges.push((node, (node * 37 + step * 97) % node_count));
        }
    }
    edges
}

fn distance_checksum(distances: &[Option<usize>]) -> f64 {
    distances
        .iter()
        .enumerate()
        .map(|(idx, distance)| (idx as f64 + 1.0) * distance.unwrap_or(0) as f64)
        .sum()
}

fn main() {
    let node_count = 5_000;
    let graph = Graph::from_edges(node_count, edges(node_count), false).unwrap();
    let warmup = graph.bfs_distances(0).unwrap();
    let checksum = distance_checksum(&warmup);
    let millis = bench_median_ms(11, || {
        let distances = graph.bfs_distances(0).unwrap();
        std::hint::black_box(distances);
    });
    println!(
        "{{\"engine\":\"graphrust\",\"cases\":[{{\"name\":\"bfs_undirected_5000_30000\",\"millis\":{millis:.6},\"checksum\":{checksum:.6}}}]}}"
    );
}
