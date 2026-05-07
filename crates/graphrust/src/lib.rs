use std::collections::VecDeque;
use std::error::Error;
use std::fmt;
use std::time::{Duration, Instant};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GraphError {
    EmptyGraph,
    NodeOutOfBounds { node: usize, node_count: usize },
}

impl fmt::Display for GraphError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyGraph => write!(f, "graph requires at least one node"),
            Self::NodeOutOfBounds { node, node_count } => {
                write!(f, "node {node} is outside graph with {node_count} nodes")
            }
        }
    }
}

impl Error for GraphError {}

pub type Result<T> = std::result::Result<T, GraphError>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Graph {
    directed: bool,
    offsets: Vec<usize>,
    neighbors: Vec<usize>,
}

impl Graph {
    pub fn from_edges(
        node_count: usize,
        edges: impl IntoIterator<Item = (usize, usize)>,
        directed: bool,
    ) -> Result<Self> {
        if node_count == 0 {
            return Err(GraphError::EmptyGraph);
        }
        let mut degrees = vec![0usize; node_count];
        let mut edge_list = Vec::new();
        for (src, dst) in edges {
            check_node(src, node_count)?;
            check_node(dst, node_count)?;
            degrees[src] += 1;
            edge_list.push((src, dst));
            if !directed && src != dst {
                degrees[dst] += 1;
                edge_list.push((dst, src));
            }
        }

        let mut offsets = Vec::with_capacity(node_count + 1);
        offsets.push(0);
        for degree in &degrees {
            offsets.push(offsets.last().copied().unwrap() + *degree);
        }

        let mut cursor = offsets[..node_count].to_vec();
        let mut neighbors = vec![0usize; edge_list.len()];
        for (src, dst) in edge_list {
            let idx = cursor[src];
            neighbors[idx] = dst;
            cursor[src] += 1;
        }

        Ok(Self {
            directed,
            offsets,
            neighbors,
        })
    }

    pub fn node_count(&self) -> usize {
        self.offsets.len() - 1
    }

    pub fn edge_count(&self) -> usize {
        if self.directed {
            self.neighbors.len()
        } else {
            self.neighbors.len() / 2
        }
    }

    pub fn is_directed(&self) -> bool {
        self.directed
    }

    pub fn neighbors(&self, node: usize) -> Result<&[usize]> {
        check_node(node, self.node_count())?;
        Ok(self.neighbors_unchecked(node))
    }

    fn neighbors_unchecked(&self, node: usize) -> &[usize] {
        &self.neighbors[self.offsets[node]..self.offsets[node + 1]]
    }

    pub fn bfs_distances(&self, start: usize) -> Result<Vec<Option<usize>>> {
        check_node(start, self.node_count())?;
        let mut distances = vec![None; self.node_count()];
        let mut queue = VecDeque::new();
        distances[start] = Some(0);
        queue.push_back(start);

        while let Some(node) = queue.pop_front() {
            let next_distance = distances[node].expect("queued nodes have distance") + 1;
            for &neighbor in self.neighbors_unchecked(node) {
                if distances[neighbor].is_none() {
                    distances[neighbor] = Some(next_distance);
                    queue.push_back(neighbor);
                }
            }
        }

        Ok(distances)
    }

    pub fn connected_components(&self) -> Vec<Vec<usize>> {
        let mut seen = vec![false; self.node_count()];
        let mut components = Vec::new();
        for start in 0..self.node_count() {
            if seen[start] {
                continue;
            }
            let mut component = Vec::new();
            let mut queue = VecDeque::new();
            seen[start] = true;
            queue.push_back(start);
            while let Some(node) = queue.pop_front() {
                component.push(node);
                for &neighbor in self.neighbors_unchecked(node) {
                    if !seen[neighbor] {
                        seen[neighbor] = true;
                        queue.push_back(neighbor);
                    }
                }
            }
            components.push(component);
        }
        components
    }

    pub fn pagerank(&self, iterations: usize, damping: f64) -> Vec<f64> {
        let n = self.node_count();
        let base = (1.0 - damping) / n as f64;
        let mut ranks = vec![1.0 / n as f64; n];
        let mut next = vec![0.0; n];

        for _ in 0..iterations {
            next.fill(base);
            let mut dangling = 0.0;
            for (node, &rank) in ranks.iter().enumerate() {
                let neighbors = self.neighbors_unchecked(node);
                if neighbors.is_empty() {
                    dangling += rank;
                    continue;
                }
                let share = damping * rank / neighbors.len() as f64;
                for &neighbor in neighbors {
                    next[neighbor] += share;
                }
            }
            if dangling != 0.0 {
                let share = damping * dangling / n as f64;
                for rank in &mut next {
                    *rank += share;
                }
            }
            std::mem::swap(&mut ranks, &mut next);
        }

        ranks
    }
}

fn check_node(node: usize, node_count: usize) -> Result<()> {
    if node >= node_count {
        Err(GraphError::NodeOutOfBounds { node, node_count })
    } else {
        Ok(())
    }
}

pub fn median_duration(mut samples: Vec<Duration>) -> Duration {
    samples.sort_unstable();
    samples[samples.len() / 2]
}

pub fn bench_median_ms<F>(rounds: usize, mut f: F) -> f64
where
    F: FnMut(),
{
    let mut samples = Vec::with_capacity(rounds);
    for _ in 0..rounds {
        let start = Instant::now();
        f();
        samples.push(start.elapsed());
    }
    median_duration(samples).as_secs_f64() * 1_000.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builds_undirected_csr_graph() {
        let graph = Graph::from_edges(4, [(0, 1), (0, 2), (2, 3)], false).unwrap();
        assert_eq!(graph.node_count(), 4);
        assert_eq!(graph.edge_count(), 3);
        assert_eq!(graph.neighbors(0).unwrap(), &[1, 2]);
        assert_eq!(graph.neighbors(2).unwrap(), &[0, 3]);
    }

    #[test]
    fn computes_bfs_distances() {
        let graph = Graph::from_edges(5, [(0, 1), (1, 2), (0, 3)], false).unwrap();
        assert_eq!(
            graph.bfs_distances(0).unwrap(),
            vec![Some(0), Some(1), Some(2), Some(1), None]
        );
    }

    #[test]
    fn computes_connected_components() {
        let graph = Graph::from_edges(6, [(0, 1), (1, 2), (4, 5)], false).unwrap();
        assert_eq!(
            graph.connected_components(),
            vec![vec![0, 1, 2], vec![3], vec![4, 5]]
        );
    }

    #[test]
    fn pagerank_is_normalized() {
        let graph = Graph::from_edges(3, [(0, 1), (1, 2), (2, 0), (2, 1)], true).unwrap();
        let ranks = graph.pagerank(25, 0.85);
        let total = ranks.iter().sum::<f64>();
        assert!((total - 1.0).abs() < 1e-12);
        assert!(ranks[1] > ranks[0]);
    }

    #[test]
    fn rejects_out_of_bounds_edges() {
        let err = Graph::from_edges(2, [(0, 2)], false).unwrap_err();
        assert_eq!(
            err,
            GraphError::NodeOutOfBounds {
                node: 2,
                node_count: 2
            }
        );
    }
}
