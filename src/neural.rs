use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;
use std::collections::HashSet;
use std::time::Instant;
// input
// [0,1,0,1,1,1]
// []

#[derive(Debug, PartialEq)]
pub struct Simulation {
    pub networks: Vec<NeuralNetwork>,
    rng: StdRng,
    /// The input layer for each network
    pub input_size: usize,
    pub output_size: usize,
}

impl Simulation {
    pub fn new(
        seed: u64,
        num_networks: usize,
        input_size: usize,
        output_size: usize,
    ) -> Simulation {
        let mut sim = Simulation {
            rng: StdRng::seed_from_u64(seed),
            networks: Vec::with_capacity(num_networks),
            input_size,
            output_size,
        };
        sim.init();
        sim
    }

    fn init(&mut self) {
        for num in 0..self.networks.capacity() {
            let mut net = NeuralNetwork {
                num,
                layers: Vec::new(),
                outputs: Vec::with_capacity(self.output_size),
            };
            net.layers.push(Vec::with_capacity(self.input_size));
            net.generate(&mut self.rng, self.output_size);
            self.networks.push(net);
        }
    }
}

#[derive(Debug, PartialEq)]
pub struct NeuralNetwork {
    pub num: usize,
    pub layers: Vec<Vec<Node>>,
    pub outputs: Vec<Node>,
}

impl NeuralNetwork {
    pub fn update(&mut self, input: &[f32]) -> Vec<f32> {
        self.layers[0].extend(input.iter().map(|x| {
            let mut n = Node::default();
            n.value = *x;
            n
        }));

        self.propagate();

        self.outputs.iter().map(|x| x.value).collect()
    }

    fn generate<R: Rng>(&mut self, rng: &mut R, output_size: usize) {
        let now = Instant::now();
        let layers = &mut self.layers;
        let num_layers = rng.gen_range(1..=5);
        for _ in 0..num_layers {
            let num_nodes = rng.gen_range(3..100);
            let new_layer = Vec::new();
            layers.push(new_layer);
            for _ in 0..num_nodes {
                let num_refs = rng.gen_range(1..10);
                let mut node = Node::default();
                node.bias = Some(rng.gen_range(-1.0..=1.0));
                for _ in 0..num_refs {
                    // Get a new layer to reference, potentially self layer
                    let layer_idx =
                        if layers[layers.len() - 1].is_empty() || rng.gen_range(0..100) > 90 {
                            // First new node of the layer, can't add a reference to self
                            // just yet, or prioritized previous layers
                            rng.gen_range(0..=layers.len() - 1)
                        } else {
                            rng.gen_range(0..layers.len())
                        };

                    let to_ref = &layers[layer_idx];
                    let node_idx = if to_ref.is_empty() {
                        0
                    } else {
                        rng.gen_range(0..to_ref.len())
                    };
                    let refs = node.references.get_or_insert(Vec::new());
                    refs.push(Reference {
                        layer: layer_idx,
                        index: node_idx,
                        weight: rng.gen_range(-1.0..1.0),
                    });
                }
                layers.last_mut().unwrap().push(node);
            }
        }
        dbg!(now.elapsed());
        self.generate_output_layer(output_size, rng);
        dbg!(now.elapsed());
    }

    fn generate_output_layer<R: Rng>(&mut self, size: usize, rng: &mut R) {
        let mut output = Vec::with_capacity(size);
        let mut orphan_nodes = HashSet::new();
        let mut nodes_with_ref = HashSet::new();
        // Find all nodes which aren't referred to
        for (i, lay) in self.layers.iter().enumerate() {
            for (j, node) in lay.iter().enumerate() {
                orphan_nodes.insert((i, j));
                if let Some(refs) = &node.references {
                    for refi in refs {
                        nodes_with_ref.insert((refi.layer, refi.index));
                    }
                }
            }
        }
        orphan_nodes.retain(|node| !nodes_with_ref.contains(node));
        let mut orphan_nodes = orphan_nodes
            .into_iter()
            .map(|x| (x, false))
            .collect::<Vec<_>>();
        for _ in 0..size {
            let mut node = Node::default();
            for orph in orphan_nodes.iter_mut() {
                let prob = 100 / size + 1;
                if rng.gen_range(0..100) <= prob {
                    // Semi evenly distribute the orphans, letting output nodes
                    // refer to the same node, potentially leaving nodes still orphaned

                    // used the node
                    orph.1 = true;
                    let addrefs = node.references.get_or_insert(Vec::new());
                    addrefs.push(Reference {
                        layer: orph.0 .0,
                        index: orph.0 .1,
                        weight: rng.gen_range(-1.0..=1.0),
                    });
                }
            }
            // Weakly bias the node
            node.bias = Some(rng.gen_range(-0.3..=0.3));
            output.push(node);
        }

        // cull anything that was used already
        orphan_nodes.retain(|x| !x.1);

        for orph in orphan_nodes {
            let len = output.len();
            // Add the remaining orphans to a random node
            if let Some(node) = output.get_mut(rng.gen_range(0..len)) {
                let addrefs = node.references.get_or_insert(Vec::new());
                addrefs.push(Reference {
                    layer: orph.0 .0,
                    index: orph.0 .1,
                    weight: rng.gen_range(-1.0..=1.0),
                });
            }
        }

        self.outputs = output;
    }

    fn propagate(&mut self) {
        assert!(
            !self.layers[0].is_empty(),
            "Cannot propagate because input is empty!"
        );
        for i in 1..self.layers.len() {
            for j in 0..self.layers[i].len() {
                let node = &self.layers[i][j];
                self.layers[i][j].value = self.compute_v(node);
            }
        }
        for ii in 0..self.outputs.len() {
            let node = &self.outputs[ii];
            self.outputs[ii].value = self.compute_v(node);
        }
    }

    fn compute_v(&self, node: &Node) -> f32 {
        let mut v = 0.;
        if let Some(references) = &node.references {
            for refi in references {
                let other = &self.layers[refi.layer][refi.index];
                v += other.value * refi.weight;
            }
        }
        if let Some(bias) = node.bias {
            v += bias;
        }
        v.clamp(-1., 1.)
    }
}

#[derive(Clone, Default, Debug, PartialEq)]
pub struct Node {
    pub references: Option<Vec<Reference>>,
    pub bias: Option<f32>,

    pub value: f32,
}

impl From<f32> for Node {
    fn from(value: f32) -> Self {
        Self {
            references: None,
            bias: None,
            value,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Reference {
    pub layer: usize,
    pub index: usize,
    pub weight: f32,
}
