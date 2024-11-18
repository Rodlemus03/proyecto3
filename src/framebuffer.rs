use nalgebra_glm::{Vec2, Vec3, Vec4};
use crate::color::Color;
use image::RgbaImage;
use nalgebra_glm::Mat4;

pub struct Framebuffer {
    pub width: usize,
    pub height: usize,
    pub buffer: Vec<u32>,
    pub zbuffer: Vec<f32>,
    background_color: u32,
    current_color: u32,
}

impl Framebuffer {
    pub fn new(width: usize, height: usize) -> Self {
        Framebuffer {
            width,
            height,
            buffer: vec![0; width * height],
            zbuffer: vec![f32::INFINITY; width * height],
            background_color: 0x000000,
            current_color: 0xFFFFFF
        }
    }

    pub fn clear(&mut self) {
        for pixel in self.buffer.iter_mut() {
            *pixel = self.background_color;
        }
        for depth in self.zbuffer.iter_mut() {
            *depth = f32::INFINITY;
        }
    }

    pub fn point(&mut self, x: usize, y: usize, depth: f32) {
        if x < self.width && y < self.height {
            let index = y * self.width + x;

            if self.zbuffer[index] > depth {
                self.buffer[index] = self.current_color;
                self.zbuffer[index] = depth;
            }
        }
    }

    pub fn set_background_color(&mut self, color: u32) {
        self.background_color = color;
    }

    pub fn set_current_color(&mut self, color: u32) {
        self.current_color = color;
    }

    pub fn textured_triangle(
        &mut self,
        v0: Vec3,          
        v1: Vec3,          
        v2: Vec3,          
        uv0: Vec2,        
        uv1: Vec2,         
        uv2: Vec2,        
        texture: &image::RgbaImage, 
    ) {
        let min_x = v0.x.min(v1.x).min(v2.x).max(0.0) as i32;
        let max_x = v0.x.max(v1.x).max(v2.x).min(self.width as f32) as i32;
        let min_y = v0.y.min(v1.y).min(v2.y).max(0.0) as i32;
        let max_y = v0.y.max(v1.y).max(v2.y).min(self.height as f32) as i32;

        let edge1 = v1 - v0;
        let edge2 = v2 - v0;
        let denominator = edge1.x * edge2.y - edge1.y * edge2.x;

        for y in min_y..=max_y {
            for x in min_x..=max_x {
                let pixel_pos = Vec3::new(x as f32, y as f32, 0.0);
                let edge_to_pixel = pixel_pos - v0;
                let u = (edge_to_pixel.x * edge2.y - edge_to_pixel.y * edge2.x) / denominator;
                let v = (edge1.x * edge_to_pixel.y - edge1.y * edge_to_pixel.x) / denominator;
                let w = 1.0 - u - v;

                if u >= 0.0 && v >= 0.0 && w >= 0.0 {
                    let uv = uv0 * w + uv1 * u + uv2 * v;
                    let tex_x = (uv.x * texture.width() as f32) as u32 % texture.width();
                    let tex_y = (uv.y * texture.height() as f32) as u32 % texture.height();
                    let tex_color = texture.get_pixel(tex_x, tex_y);

                    let color = Color::new(tex_color[0], tex_color[1], tex_color[2]).to_hex();
                    
                    self.point(x as usize, y as usize, pixel_pos.z);
                }
            }
        }
    }

    pub fn draw_skybox_face(
        &mut self,
        v0: Vec3,
        v1: Vec3,
        v2: Vec3,
        v3: Vec3,
        texture: &image::RgbaImage,
        view_matrix: &Mat4,
        projection_matrix: &Mat4,
    ) {
        let vertices = [v0, v1, v2, v3];
        let mut projected_vertices = Vec::with_capacity(4);
    
        for vertex in &vertices {
            let transformed_vertex = projection_matrix * view_matrix * Vec4::new(vertex.x, vertex.y, vertex.z, 1.0);
    
            let projected_vertex = Vec3::new(
                transformed_vertex.x / transformed_vertex.w,
                transformed_vertex.y / transformed_vertex.w,
                transformed_vertex.z / transformed_vertex.w,
            );
    
            projected_vertices.push(projected_vertex);
        }
    
        let uvs = [
            Vec2::new(0.0, 0.0),
            Vec2::new(1.0, 0.0),
            Vec2::new(1.0, 1.0),
            Vec2::new(0.0, 1.0),
        ];
    
        self.textured_triangle(
            projected_vertices[0],
            projected_vertices[1],
            projected_vertices[2],
            uvs[0],
            uvs[1],
            uvs[2],
            texture,
        );
    
        self.textured_triangle(
            projected_vertices[0],
            projected_vertices[2],
            projected_vertices[3],
            uvs[0],
            uvs[2],
            uvs[3],
            texture,
        );
    }
    
}