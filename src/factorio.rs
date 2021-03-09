use bmp::Image;
use bmp::Pixel;

const WATER_COLOR: Pixel = Pixel {
    r: 45,
    g: 94,
    b: 127,
};

#[derive(Debug)]
pub struct DrivableTileMap {
    tiles: Vec<Vec<bool>>,
}

impl DrivableTileMap {
    pub fn new(capacity: usize, inner_capacity: usize) -> Self {
        DrivableTileMap {
            tiles: vec![vec![false; inner_capacity]; capacity],
        }
    }

    pub fn drivable(&self, x: usize, y: usize) -> Option<bool> {
        Some(*self.tiles.get(x)?.get(y)?)
    }
}

impl std::fmt::Display for DrivableTileMap {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::result::Result<(), std::fmt::Error> {
        for (x, row) in self.tiles.iter().enumerate() {
            write!(f, "{}:", x);
            for (_y, drivable) in row.iter().enumerate() {
                write!(f, "{}", if *drivable { "1" } else { "0" });
            }
        }
        writeln!(f);
        Ok(())
    }
}

impl From<&Image> for DrivableTileMap {
    fn from(bmp: &Image) -> Self {
        let mut map = DrivableTileMap::new(bmp.get_width() as usize, bmp.get_height() as usize);
        for (x, y) in bmp.coordinates() {
            map.tiles[x as usize][y as usize] = bmp.get_pixel(x, y) != WATER_COLOR;
        }
        map
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_load_file() {
        let b = bmp::open("example.bmp").unwrap();
        DrivableTileMap::from(&b);
    }
}
