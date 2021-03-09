use crate::factorio::DrivableTileMap;
use crate::neural::NeuralNetwork;
use crate::neural::Simulation;
use rcon::Builder;
use rcon::Connection;
use std::f64::consts::TAU;
use std::fs::File;
use std::io::Write;

mod neural;

mod factorio;

struct World {
    tiles: DrivableTileMap,
}

/// Information about a Car
struct Car {
    position: Position,
    speed: f64,
    /// The orientation of the car in Factorio units
    orientation: f64,
    fuel: f64,
}

impl Car {
    /// The orientation of the car in radians
    fn radians(&self) -> f64 {
        ((self.orientation * -1.0) * TAU + TAU * 1.25) % TAU
    }

    /// Returns a list of the tile positions under this car's vision, sorted
    /// by proximity to the car.
    fn vision_positions(&self) -> Vec<(usize, usize)> {
        let mut v = Vec::new();
        let vis_left_angle = self.radians() + 15_f64.to_radians();
        let vis_right_angle = self.radians() - 15_f64.to_radians();

        v
    }
}

impl Car {
    fn score(&self) -> f64 {
        self.position.dist(&Position::default()) * self.fuel
    }
}

/// The position of a thing in the game world
#[derive(Default)]
struct Position {
    x: f64,
    y: f64,
}

impl Position {
    fn dist(&self, other: &Position) -> f64 {
        ((self.x - other.x).powf(2.0) + (self.y - other.y).powf(2.0)).sqrt()
    }
}

/// RidingState
struct RidingState {
    acceleration: Acceleration,
    direction: Direction,
}

enum Acceleration {
    Nothing,
    Accelerating,
    Braking,
    Reversing,
}

impl From<f32> for Acceleration {
    fn from(d: f32) -> Acceleration {
        if d < -1. / 2. {
            Acceleration::Nothing
        } else if d < 0. {
            Acceleration::Accelerating
        } else if d < 1. / 2. {
            Acceleration::Braking
        } else if d < 1. {
            Acceleration::Reversing
        } else {
            unreachable!();
        }
    }
}

enum Direction {
    Straight,
    Right,
    Left,
}

impl From<f32> for Direction {
    fn from(d: f32) -> Direction {
        if d < -1. / 3. {
            Direction::Left
        } else if d < 1. / 3. {
            Direction::Straight
        } else if d < 1. {
            Direction::Right
        } else {
            unreachable!();
        }
    }
}

fn convert_output(out: &[f32; 2]) -> RidingState {
    RidingState {
        acceleration: out[0].into(),
        direction: out[1].into(),
    }
}

fn init_serialize_car_fn(conn: &mut Connection) -> Result<(), Box<dyn std::error::Error>> {
    conn.cmd(
        r#"
    /sc function serialize_car(ent)

        end
    "#,
    )?;
    Ok(())
}

fn create_pool(conn: &mut Connection) -> Result<(), Box<dyn std::error::Error>> {
    let resp = conn.cmd("/sc pool = {}");
    Ok(())
}

fn spawn_car(conn: &mut Connection) -> Result<String, Box<dyn std::error::Error>> {
    let resp = conn.cmd(
        r#"
        /sc local ent = game.surfaces["nauvis"].create_entity({name = "car", position={0,0}, force="player"})
            ent.insert({name = "solid-fuel", count = 1})
            table.insert(pool, ent)
            rcon.print(serialize_car(ent))
        "#)?;
    println!("{}", resp);
    Ok(resp)
}

/// Exports a neural network into a graphviz dot file.
fn export(net: &NeuralNetwork, f: &mut File) -> Result<(), Box<dyn std::error::Error>> {
    let mut s = String::new();
    s.push_str("digraph {\n");
    for (i, layer) in net.layers.iter().enumerate() {
        for (j, node) in layer.iter().enumerate() {
            s.push_str(&format!(
                "\t{}{} [label=\"{},{}{}\"]\n",
                i, j, i, j, node.value
            ));
            if let Some(refs) = &node.references {
                for refi in refs {
                    s.push_str(&format!("\t{}{} -> {}{}\n", i, j, refi.layer, refi.index));
                }
            }
        }
    }
    let out_lay_idx = net.layers.len();
    dbg!(out_lay_idx);
    for (ii, node) in net.outputs.iter().enumerate() {
        s.push_str(&format!(
            "\t{}{} [label=\"{},{}{}\"]\n",
            out_lay_idx, ii, out_lay_idx, ii, node.value
        ));
        if let Some(refs) = &node.references {
            for refi in refs {
                s.push_str(&format!(
                    "\t{}{} -> {}{}\n",
                    out_lay_idx, ii, refi.layer, refi.index
                ));
            }
        }
    }
    s.push_str("\tsubgraph subs {");
    s.push_str("\t\trank=\"same\"");
    for ii in 0..net.outputs.len() {
        s.push_str(&format!("\t\t{}{}", out_lay_idx, ii));
    }
    s.push_str("\t}");

    s.push_str("}\n");
    f.write_all(&s.as_bytes())?;
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let now = std::time::Instant::now();
    let mut sim0 = Simulation::new(1, 3000, 2, 2);
    assert!(!sim0.networks[0].outputs.is_empty());
    let mut dot = File::create("testing.dot")?;
    //export(&sim0.networks[0], &mut dot)?;
    eprintln!("{:?}", now.elapsed());
    let v = &[0.1, 0.2];
    for net in &mut sim0.networks {
        net.update(v);
    }
    let mut dot2 = File::create("testing2.dot")?;
    //export(&sim0.networks[0], &mut dot2)?;
    eprintln!("{:?}", now.elapsed());
    let mut conn = Builder::new()
        .enable_factorio_quirks(true)
        .connect("127.0.0.1:7777", "zzz")?;
    create_pool(&mut conn)?;
    spawn_car(&mut conn)?;
    spawn_car(&mut conn)?;
    Ok(())
}

/*
Self - the car
tick
Position x y
Speed m/s
orientation
fuel
Score: distance from 0, 0

World
[] Obstacles
Each: position

Outputs
1 or 0
W
a
s
d











*/
