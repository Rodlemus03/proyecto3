use std::collections::HashMap;
use std::sync::Arc;
use image::RgbaImage;

pub struct TextureManager {
    textures: HashMap<String, Arc<RgbaImage>>,
}

impl TextureManager {
    pub fn new() -> Self {
        TextureManager {
            textures: HashMap::new(),
        }
    }

    pub fn cargar_textura(&mut self, name: &str, image: RgbaImage) {
        let texture = Arc::new(image);
        self.textures.insert(name.to_string(), texture);
    }

    pub fn get_textura(&self, name: &str) -> Option<Arc<RgbaImage>> {
        self.textures.get(name).cloned()
    }
}