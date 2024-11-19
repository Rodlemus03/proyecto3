use minifb::{Key, Window, WindowOptions};
use nalgebra_glm::{look_at, perspective, Mat4, Vec3};
use std::f32::consts::PI;
use std::time::Duration;

mod camera;
mod color;
mod fragment;
mod framebuffer;
mod material;
mod obj;
mod ray_intersect;
mod shaders;
mod skybox;
mod texturas;
mod triangle;
mod vertex;

use crate::color::Color;
use crate::fragment::Fragment;
use crate::material::Material;
use crate::ray_intersect::{Intersect, RayIntersect};
use crate::shaders::vertex_shader;
use crate::shaders::vertex_shader_simplex;
use crate::skybox::Skybox;
use crate::texturas::TextureManager;
use camera::Camera;
use fastnoise_lite::FastNoiseLite;
use framebuffer::Framebuffer;
use nalgebra_glm::normalize;
use noise::Simplex;
use obj::Obj;
use rayon::prelude::*;
use shaders::{
    anillos_shader, bacteria_shader, camo_shader, cellular_shader, lava_shader, planeta_gaseoso,
    shader_agua, shader_agujero_negro, shader_grupos, shader_luna, shader_puntas, shader_variado,
};
use std::sync::{Arc, Mutex};
use std::time::Instant;
use triangle::triangle;
use vertex::Vertex;

pub struct Uniforms {
    MM: Mat4,
    view_matrix: Mat4,
    projection_matrix: Mat4,
    viewport_matrix: Mat4,
    time: u32,
    noise: FastNoiseLite,
}

pub struct Uniforms_Simplex {
    MM: Mat4,
    view_matrix: Mat4,
    projection_matrix: Mat4,
    viewport_matrix: Mat4,
    time: u32,
    noise: Simplex,
}

fn ruidoSimple() -> Simplex {
    Simplex::new(100)
}

fn ruidoPerl() -> FastNoiseLite {
    let mut noise = FastNoiseLite::new();

    noise.set_noise_type(Some(fastnoise_lite::NoiseType::Perlin));

    noise.set_seed(Some(100));
    noise.set_frequency(Some(0.030));

    noise.set_fractal_type(Some(fastnoise_lite::FractalType::PingPong));
    noise.set_fractal_octaves(Some(9));
    noise.set_fractal_lacunarity(Some(1.0));
    noise.set_fractal_gain(Some(0.100));
    noise.set_fractal_ping_pong_strength(Some(9.0));
    noise
}

fn ruidoCel() -> FastNoiseLite {
    let mut noise = FastNoiseLite::new();
    noise.set_noise_type(Some(fastnoise_lite::NoiseType::Cellular));
    noise.set_seed(Some(100));
    noise.set_frequency(Some(0.080));
    noise.set_cellular_distance_function(Some(
        fastnoise_lite::CellularDistanceFunction::EuclideanSq,
    ));
    noise.set_cellular_return_type(Some(fastnoise_lite::CellularReturnType::Distance2Div));
    noise.set_cellular_jitter(Some(1.0));
    noise.set_fractal_type(Some(fastnoise_lite::FractalType::FBm));
    noise.set_fractal_octaves(Some(9));
    noise.set_fractal_lacunarity(Some(1.0));
    noise.set_fractal_gain(Some(0.3));
    noise
}

fn ruidoCelBac() -> FastNoiseLite {
    let mut noise = FastNoiseLite::new();
    noise.set_noise_type(Some(fastnoise_lite::NoiseType::Cellular));
    noise.set_seed(Some(1337));
    noise.set_frequency(Some(0.010));
    noise.set_cellular_distance_function(Some(
        fastnoise_lite::CellularDistanceFunction::EuclideanSq,
    ));
    noise.set_cellular_return_type(Some(fastnoise_lite::CellularReturnType::Distance2Mul));
    noise.set_cellular_jitter(Some(1.0));
    noise.set_fractal_type(Some(fastnoise_lite::FractalType::PingPong));
    noise.set_fractal_octaves(Some(3));
    noise.set_fractal_lacunarity(Some(2.0));
    noise.set_fractal_gain(Some(1.0));
    noise.set_fractal_ping_pong_strength(Some(7.0));
    noise
}

fn ruidoCelAN() -> FastNoiseLite {
    let mut noise = FastNoiseLite::new();
    noise.set_noise_type(Some(fastnoise_lite::NoiseType::Perlin));
    noise.set_seed(Some(100));
    noise.set_frequency(Some(0.030));
    noise.set_fractal_type(Some(fastnoise_lite::FractalType::PingPong));
    noise.set_fractal_octaves(Some(9));
    noise.set_fractal_lacunarity(Some(1.0));
    noise.set_fractal_gain(Some(1.0));
    noise.set_fractal_weighted_strength(Some(3.0));
    noise.set_fractal_ping_pong_strength(Some(10.0));
    noise
}

fn ruidoCamo() -> FastNoiseLite {
    let mut noise = FastNoiseLite::new();
    noise.set_noise_type(Some(fastnoise_lite::NoiseType::OpenSimplex2));
    noise.set_seed(Some(1337));
    noise.set_frequency(Some(0.010));
    noise.set_fractal_type(Some(fastnoise_lite::FractalType::Ridged));
    noise.set_fractal_octaves(Some(9));
    noise.set_fractal_lacunarity(Some(5.0));
    noise.set_fractal_gain(Some(1.0));
    noise.set_fractal_weighted_strength(Some(7.0));
    noise
}

fn ruidoVariado() -> FastNoiseLite {
    let mut noise = FastNoiseLite::new();
    noise.set_noise_type(Some(fastnoise_lite::NoiseType::Cellular));
    noise.set_seed(Some(100));
    noise.set_frequency(Some(0.030));
    noise.set_fractal_type(Some(fastnoise_lite::FractalType::FBm));
    noise.set_fractal_octaves(Some(9));
    noise.set_fractal_lacunarity(Some(1.0));
    noise.set_fractal_gain(Some(1.0));
    noise.set_fractal_weighted_strength(Some(3.0));
    noise.set_cellular_distance_function(Some(
        fastnoise_lite::CellularDistanceFunction::EuclideanSq,
    ));
    noise.set_cellular_return_type(Some(fastnoise_lite::CellularReturnType::Distance2Div));
    noise.set_cellular_jitter(Some(1.0));
    noise
}

fn ruidoGrupos() -> FastNoiseLite {
    let mut noise = FastNoiseLite::new();
    noise.set_noise_type(Some(fastnoise_lite::NoiseType::Cellular));
    noise.set_seed(Some(1337));
    noise.set_frequency(Some(0.030));
    noise.set_cellular_distance_function(Some(fastnoise_lite::CellularDistanceFunction::Hybrid));
    noise.set_cellular_return_type(Some(fastnoise_lite::CellularReturnType::Distance2Sub));
    noise.set_cellular_jitter(Some(2.0));
    noise.set_fractal_type(Some(fastnoise_lite::FractalType::PingPong));
    noise.set_fractal_octaves(Some(3));
    noise.set_fractal_lacunarity(Some(2.0));
    noise.set_fractal_gain(Some(0.5));
    noise.set_fractal_ping_pong_strength(Some(1.0));
    noise
}

fn ruidoCelPunt() -> FastNoiseLite {
    let mut noise = FastNoiseLite::new();
    noise.set_noise_type(Some(fastnoise_lite::NoiseType::Cellular));
    noise.set_seed(Some(1337));
    noise.set_frequency(Some(0.030));
    noise.set_cellular_distance_function(Some(fastnoise_lite::CellularDistanceFunction::Manhattan));
    noise.set_cellular_return_type(Some(fastnoise_lite::CellularReturnType::Distance));
    noise.set_cellular_jitter(Some(1.0));
    noise
}

pub fn castRay(
    ray_origin: &Vec3,
    ray_direction: &Vec3,
    objects: &[Box<dyn RayIntersect>],
    color_fondo: &Color,
) -> Color {
    let mut intersect = Intersect::empty();
    let mut zbuffer = f32::INFINITY;

    for object in objects {
        let tmp = object.ray_intersect(ray_origin, ray_direction);
        if tmp.is_intersecting && tmp.distance < zbuffer {
            zbuffer = tmp.distance;
            intersect = tmp;
        }
    }

    if !intersect.is_intersecting {
        return color_fondo.clone();
    }

    let mut color = intersect.material.diffuse.clone();
    if let Some(ref textura) = intersect.material.textura {
        color = intersect
            .material
            .get_diffuse_color(intersect.u, intersect.v);
    }

    color
}

fn render_skybox(
    framebuffer: &mut Framebuffer,
    objects: &[Box<dyn RayIntersect>],
    camera: &Camera,
    color_fondo: &Color,
) {
    let width = framebuffer.width as f32;
    let height = framebuffer.height as f32;
    let aspect_ratio = width / height;
    let fov = PI / 3.0;
    let perspective_scale = (fov * 0.5).tan();

    framebuffer
        .buffer
        .par_chunks_mut(framebuffer.width)
        .enumerate()
        .for_each(|(y, row)| {
            let screen_y = -(2.0 * y as f32) / height + 1.0;
            let screen_y = screen_y * perspective_scale;

            for x in 0..framebuffer.width {
                let screen_x = (2.0 * x as f32) / width - 1.0;
                let screen_x = screen_x * aspect_ratio * perspective_scale;

                let ray_direction = normalize(&Vec3::new(screen_x, screen_y, -1.0));
                let rotated_direction = camera.base_change(&ray_direction);

                let pixel_color = castRay(&camera.eye, &rotated_direction, objects, color_fondo);
                row[x] = pixel_color.to_hex();
            }
        });
}

fn main() {
    let window_width = 1000;
    let window_height = 800;
    let framebuffer_width = 1000;
    let framebuffer_height = 800;

    let mut framebuffer = Framebuffer::new(framebuffer_width, framebuffer_height);
    let mut window = Window::new(
        "Proyecto SS",
        window_width,
        window_height,
        WindowOptions::default(),
    )
    .unwrap();

    window.set_position(500, 500);
    window.update();

    let mut textMan = TextureManager::new();
    let fondoImagen = image::open("assets/cielo.jpg").unwrap().into_rgba8();
    textMan.cargar_textura("cielo", fondoImagen);
    let textFondo = textMan.get_textura("cielo");

    let cielo = Material::new(
        Color::new(255, 255, 255),
        0.0,
        [0.0, 0.0],
        textFondo.clone(),
        None,
    );

    framebuffer.set_background_color(0x333355);

    let rotation = Vec3::new(0.0, 0.0, 0.0);
    let rotation_anillos = Vec3::new(PI / 4.0, 0.0, 0.0);
    let scale = 1.0f32;

    let mut camera = Camera::new(
        Vec3::new(50.0, 0.0, 150.0),
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(0.0, 1.0, 0.0),
    );

    let esfera = Obj::load("assets/modelos/sphere.obj").expect("Failed to load sphere.obj");
    let arrayEsfera = esfera.get_vertex_array();

    let anillos = Obj::load("assets/modelos/anillos.obj").expect("Failed to load anillos.obj");
    let arrayAnillos = anillos.get_vertex_array();

    let mut time = 0;
    let mut shader_actual = 1;

    const FRAME_DELAY: Duration = Duration::from_millis(16);

    while window.is_open() {
        let start_time = Instant::now();
        if window.is_key_down(Key::Escape) {
            break;
        }

        time += 1;

        framebuffer.clear();

        let view_matrix = vistaM(camera.eye, camera.center, camera.up);
        let projection_matrix = Mperspec(window_width as f32, window_height as f32);
        let viewport_matrix =
            create_viewport_matrix(framebuffer_width as f32, framebuffer_height as f32);

        let centro = Vec3::new(0.0, 0.0, 0.0);
        let grande = 10000.0;
        let color_fondo = Color::new(135, 206, 235);

        let mut objects: Vec<Box<dyn RayIntersect>> = vec![Box::new(Skybox {
            center: centro,
            size: grande,
            materials: [
                cielo.clone(),
                cielo.clone(),
                cielo.clone(),
                cielo.clone(),
                cielo.clone(),
                cielo.clone(),
            ],
        })];

        render_skybox(&mut framebuffer, &objects, &camera, &color_fondo);

        let radios_orbitales = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0];
        let velocidades_orbitales = vec![
            0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009,
        ];

        let velocidades_rotacion = vec![
            0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.056,
        ];

        let Rp_1 = Vec3::new(0.0, time as f32 * velocidades_rotacion[0], 0.0);
        let Rp_2 = Vec3::new(0.0, time as f32 * velocidades_rotacion[1], 0.0);
        let Rp_3 = Vec3::new(0.0, time as f32 * velocidades_rotacion[2], 0.0);
        let Rp_4 = Vec3::new(0.0, time as f32 * velocidades_rotacion[3], 0.0);
        let Rp_5 = Vec3::new(0.0, time as f32 * velocidades_rotacion[4], 0.0);
        let Rp_6 = Vec3::new(0.0, time as f32 * velocidades_rotacion[5], 0.0);
        let Rp_7 = Vec3::new(0.0, time as f32 * velocidades_rotacion[6], 0.0);
        let Rp_8 = Vec3::new(0.0, time as f32 * velocidades_rotacion[7], 0.0);
        let Rp_9 = Vec3::new(0.0, time as f32 * velocidades_rotacion[8], 0.0);
        let Rp_10 = Vec3::new(0.0, time as f32 * velocidades_rotacion[9], 0.0);

        let MM_1 = modeloMat(
            Vec3::new(
                radios_orbitales[1] * (time as f32 * velocidades_orbitales[1]).cos(),
                radios_orbitales[1] * (time as f32 * velocidades_orbitales[1]).sin(),
                0.0,
            ),
            scale,
            Rp_1,
        );
        let MM_anillos = modeloMat(
            Vec3::new(
                radios_orbitales[1] * (time as f32 * velocidades_orbitales[1]).cos(),
                radios_orbitales[1] * (time as f32 * velocidades_orbitales[1]).sin(),
                0.0,
            ),
            scale,
            rotation_anillos,
        );
        let MM_2 = modeloMat(
            Vec3::new(
                radios_orbitales[0] * (time as f32 * velocidades_orbitales[0]).cos(),
                radios_orbitales[0] * (time as f32 * velocidades_orbitales[0]).sin(),
                0.0,
            ),
            scale,
            Rp_2,
        );
        let MM_3 = modeloMat(Vec3::new(0.0, 0.0, 0.0), scale, Rp_10);
        let MM_4 = modeloMat(
            Vec3::new(
                radios_orbitales[2] * (time as f32 * velocidades_orbitales[2]).cos(),
                radios_orbitales[2] * (time as f32 * velocidades_orbitales[2]).sin(),
                0.0,
            ),
            scale,
            Rp_3,
        );
        let MM_5 = modeloMat(
            Vec3::new(
                radios_orbitales[3] * (time as f32 * velocidades_orbitales[3]).cos(),
                radios_orbitales[3] * (time as f32 * velocidades_orbitales[3]).sin(),
                0.0,
            ),
            scale,
            Rp_4,
        );
        let MM_6 = modeloMat(
            Vec3::new(
                radios_orbitales[4] * (time as f32 * velocidades_orbitales[4]).cos(),
                radios_orbitales[4] * (time as f32 * velocidades_orbitales[4]).sin(),
                0.0,
            ),
            scale,
            Rp_5,
        );
        let MM_7 = modeloMat(
            Vec3::new(
                radios_orbitales[5] * (time as f32 * velocidades_orbitales[5]).cos(),
                radios_orbitales[5] * (time as f32 * velocidades_orbitales[5]).sin(),
                0.0,
            ),
            scale,
            Rp_6,
        );
        let MM_8 = modeloMat(
            Vec3::new(
                radios_orbitales[6] * (time as f32 * velocidades_orbitales[6]).cos(),
                radios_orbitales[6] * (time as f32 * velocidades_orbitales[6]).sin(),
                0.0,
            ),
            scale,
            Rp_7,
        );
        let MM_9 = modeloMat(
            Vec3::new(
                radios_orbitales[7] * (time as f32 * velocidades_orbitales[7]).cos(),
                radios_orbitales[7] * (time as f32 * velocidades_orbitales[7]).sin(),
                0.0,
            ),
            scale,
            Rp_8,
        );
        let MM_10 = modeloMat(
            Vec3::new(
                radios_orbitales[8] * (time as f32 * velocidades_orbitales[8]).cos(),
                radios_orbitales[8] * (time as f32 * velocidades_orbitales[8]).sin(),
                0.0,
            ),
            scale,
            Rp_9,
        );

        fn create_uniforms(
            MM: Mat4,
            view_matrix: &Mat4,
            projection_matrix: &Mat4,
            viewport_matrix: &Mat4,
            time: u32,
            noise: FastNoiseLite,
        ) -> Uniforms {
            Uniforms {
                MM,
                view_matrix: view_matrix.clone(),
                projection_matrix: projection_matrix.clone(),
                viewport_matrix: viewport_matrix.clone(),
                time,
                noise,
            }
        }

        let uniforms_gaseoso = create_uniforms(
            MM_1,
            &view_matrix,
            &projection_matrix,
            &viewport_matrix,
            time,
            ruidoPerl(),
        );

        let uniforms_anillos = create_uniforms(
            MM_anillos,
            &view_matrix,
            &projection_matrix,
            &viewport_matrix,
            time,
            ruidoPerl(),
        );

        let uniforms_cellular_puntas = create_uniforms(
            MM_2,
            &view_matrix,
            &projection_matrix,
            &viewport_matrix,
            time,
            ruidoCelPunt(),
        );

        let uniforms_lava = create_uniforms(
            MM_3,
            &view_matrix,
            &projection_matrix,
            &viewport_matrix,
            time,
            ruidoPerl(),
        );

        let uniforms_cellular_grupos = create_uniforms(
            MM_4,
            &view_matrix,
            &projection_matrix,
            &viewport_matrix,
            time,
            ruidoGrupos(),
        );

        let uniforms_cellular = create_uniforms(
            MM_5,
            &view_matrix,
            &projection_matrix,
            &viewport_matrix,
            time,
            ruidoCel(),
        );

        let uniforms_agua = create_uniforms(
            MM_6,
            &view_matrix,
            &projection_matrix,
            &viewport_matrix,
            time,
            ruidoPerl(),
        );

        let uniforms_cellular_bacteria = create_uniforms(
            MM_7,
            &view_matrix,
            &projection_matrix,
            &viewport_matrix,
            time,
            ruidoCelBac(),
        );

        let uniforms_camo = create_uniforms(
            MM_8,
            &view_matrix,
            &projection_matrix,
            &viewport_matrix,
            time,
            ruidoCamo(),
        );

        let uniforms_cellular_agujero_negro = create_uniforms(
            MM_9,
            &view_matrix,
            &projection_matrix,
            &viewport_matrix,
            time,
            ruidoCelAN(),
        );

        let uniforms_variado = create_uniforms(
            MM_10,
            &view_matrix,
            &projection_matrix,
            &viewport_matrix,
            time,
            ruidoVariado(),
        );

        handle_input(&window, &mut camera);

        framebuffer.set_current_color(0xFFDDDD);

        renderizarSombra(&mut framebuffer, &uniforms_lava, &arrayEsfera, lava_shader);
        renderizarSombra(
            &mut framebuffer,
            &uniforms_cellular_puntas,
            &arrayEsfera,
            shader_puntas,
        );
        renderizarSombra(
            &mut framebuffer,
            &uniforms_gaseoso,
            &arrayEsfera,
            planeta_gaseoso,
        );
        renderizarSombra(
            &mut framebuffer,
            &uniforms_anillos,
            &arrayAnillos,
            anillos_shader,
        );
        renderizarSombra(
            &mut framebuffer,
            &uniforms_cellular_grupos,
            &arrayEsfera,
            shader_grupos,
        );
        renderizarSombra(
            &mut framebuffer,
            &uniforms_cellular,
            &arrayEsfera,
            cellular_shader,
        );

        let radio_orbita = 2.0;
        let velocidad_orbita = 0.02;
        let x_offset = radio_orbita * (time as f32 * velocidad_orbita).cos();
        let z_offset = radio_orbita * (time as f32 * velocidad_orbita).sin();

        let translacion_luna = Vec3::new(
            radios_orbitales[3] * (time as f32 * velocidades_orbitales[3]).cos(),
            radios_orbitales[3] * (time as f32 * velocidades_orbitales[3]).sin(),
            0.0,
        ) + Vec3::new(x_offset, 0.0, z_offset);
        let escala_luna = 0.5;
        let MM_luna = modeloMat(translacion_luna, escala_luna, rotation);

        let luna = Uniforms_Simplex {
            MM: MM_luna,
            view_matrix: view_matrix.clone(),
            projection_matrix: projection_matrix.clone(),
            viewport_matrix: viewport_matrix.clone(),
            time,
            noise: ruidoSimple(),
        };

        render_shader_simplex(&mut framebuffer, &luna, &arrayEsfera, shader_luna);
        renderizarSombra(&mut framebuffer, &uniforms_agua, &arrayEsfera, shader_agua);
        renderizarSombra(
            &mut framebuffer,
            &uniforms_cellular_bacteria,
            &arrayEsfera,
            bacteria_shader,
        );
        renderizarSombra(&mut framebuffer, &uniforms_camo, &arrayEsfera, camo_shader);
        renderizarSombra(
            &mut framebuffer,
            &uniforms_cellular_agujero_negro,
            &arrayEsfera,
            shader_agujero_negro,
        );
        renderizarSombra(
            &mut framebuffer,
            &uniforms_variado,
            &arrayEsfera,
            shader_variado,
        );

        window
            .update_with_buffer(&framebuffer.buffer, framebuffer_width, framebuffer_height)
            .unwrap();

        let elapsed = start_time.elapsed();
        if elapsed < FRAME_DELAY {
            std::thread::sleep(FRAME_DELAY - elapsed);
        }
    }
}

fn renderizarSombra(
    framebuffer: &mut Framebuffer,
    uniforms: &Uniforms,
    vertex_array: &[Vertex],
    fragment_shader_fn: fn(&Fragment, &Uniforms) -> Color,
) {
    let width = framebuffer.width;
    let height = framebuffer.height;

    let color_buffer = Arc::new(Mutex::new(vec![0; width * height]));
    let depth_buffer = Arc::new(Mutex::new(vec![f32::INFINITY; width * height]));

    let transformed_vertices: Vec<_> = vertex_array
        .par_iter()
        .map(|vertex| vertex_shader(vertex, uniforms))
        .collect();

    let triangles: Vec<_> = transformed_vertices
        .chunks(3)
        .filter(|chunk| chunk.len() == 3)
        .map(|chunk| [chunk[0].clone(), chunk[1].clone(), chunk[2].clone()])
        .collect();

    triangles.par_iter().for_each(|tri| {
        let fragments = triangle(&tri[0], &tri[1], &tri[2]);

        for fragment in fragments {
            let x = fragment.position.x as usize;
            let y = fragment.position.y as usize;

            if x < width && y < height {
                let index = y * width + x;
                let shaded_color = fragment_shader_fn(&fragment, uniforms).to_hex();

                let mut color_buffer = color_buffer.lock().unwrap();
                let mut depth_buffer = depth_buffer.lock().unwrap();

                if fragment.depth < depth_buffer[index] {
                    color_buffer[index] = shaded_color;
                    depth_buffer[index] = fragment.depth;
                }
            }
        }
    });

    let cBuff = color_buffer.lock().unwrap();
    let dBuffer = depth_buffer.lock().unwrap();

    for y in 0..height {
        for x in 0..width {
            let index = y * width + x;
            framebuffer.set_current_color(cBuff[index]);
            framebuffer.point(x, y, dBuffer[index]);
        }
    }
}

fn render_shader_simplex(
    framebuffer: &mut Framebuffer,
    uniforms: &Uniforms_Simplex,
    vertex_array: &[Vertex],
    fragment_shader_fn: fn(&Fragment, &Uniforms_Simplex) -> Color,
) {
    let mut transformed_vertices = Vec::with_capacity(vertex_array.len());
    for vertex in vertex_array {
        let transformed = vertex_shader_simplex(vertex, uniforms);
        transformed_vertices.push(transformed);
    }

    let mut triangles = Vec::new();
    for i in (0..transformed_vertices.len()).step_by(3) {
        if i + 2 < transformed_vertices.len() {
            triangles.push([
                transformed_vertices[i].clone(),
                transformed_vertices[i + 1].clone(),
                transformed_vertices[i + 2].clone(),
            ]);
        }
    }

    let mut fragments = Vec::new();
    for tri in &triangles {
        fragments.extend(triangle(&tri[0], &tri[1], &tri[2]));
    }

    for fragment in fragments {
        let x = fragment.position.x as usize;
        let y = fragment.position.y as usize;

        if x < framebuffer.width && y < framebuffer.height {
            let shaded_color = fragment_shader_fn(&fragment, &uniforms);
            let color = shaded_color.to_hex();
            framebuffer.set_current_color(color);
            framebuffer.point(x, y, fragment.depth);
        }
    }
}

fn handle_input(window: &Window, camera: &mut Camera) {
    let movement_speed = 5.0;
    let rotation_speed = PI / 50.0;

    let mut movement = Vec3::new(0.0, 0.0, 0.0);

    if window.is_key_down(Key::W) {
        movement += camera.get_forward() * movement_speed;
    }
    if window.is_key_down(Key::S) {
        movement -= camera.get_forward() * movement_speed;
    }

    if window.is_key_down(Key::A) {
        movement -= camera.get_right() * movement_speed;
    }
    if window.is_key_down(Key::D) {
        movement += camera.get_right() * movement_speed;
    }

    camera.move_center(movement);

    if window.is_key_down(Key::Left) {
        camera.rotate_y(rotation_speed);
    }
    if window.is_key_down(Key::Right) {
        camera.rotate_y(-rotation_speed);
    }
    if window.is_key_down(Key::Up) {
        camera.rotate_x(rotation_speed);
    }
    if window.is_key_down(Key::Down) {
        camera.rotate_x(-rotation_speed);
    }
}

fn modeloMat(translation: Vec3, scale: f32, rotation: Vec3) -> Mat4 {
    let (sin_x, cos_x) = rotation.x.sin_cos();
    let (sin_y, cos_y) = rotation.y.sin_cos();
    let (sin_z, cos_z) = rotation.z.sin_cos();

    let rotation_matrix_x = Mat4::new(
        1.0, 0.0, 0.0, 0.0, 0.0, cos_x, -sin_x, 0.0, 0.0, sin_x, cos_x, 0.0, 0.0, 0.0, 0.0, 1.0,
    );

    let rotation_matrix_y = Mat4::new(
        cos_y, 0.0, sin_y, 0.0, 0.0, 1.0, 0.0, 0.0, -sin_y, 0.0, cos_y, 0.0, 0.0, 0.0, 0.0, 1.0,
    );

    let rotation_matrix_z = Mat4::new(
        cos_z, -sin_z, 0.0, 0.0, sin_z, cos_z, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    );

    let rotation_matrix = rotation_matrix_z * rotation_matrix_y * rotation_matrix_x;

    let transform_matrix = Mat4::new(
        scale,
        0.0,
        0.0,
        translation.x,
        0.0,
        scale,
        0.0,
        translation.y,
        0.0,
        0.0,
        scale,
        translation.z,
        0.0,
        0.0,
        0.0,
        1.0,
    );

    transform_matrix * rotation_matrix
}

fn vistaM(eye: Vec3, center: Vec3, up: Vec3) -> Mat4 {
    look_at(&eye, &center, &up)
}

fn Mperspec(window_width: f32, window_height: f32) -> Mat4 {
    let fov = 45.0 * PI / 180.0;
    let aspect_ratio = window_width / window_height;
    let near = 0.1;
    let far = 1000.0;

    perspective(fov, aspect_ratio, near, far)
}

fn create_viewport_matrix(width: f32, height: f32) -> Mat4 {
    Mat4::new(
        width / 2.0,
        0.0,
        0.0,
        width / 2.0,
        0.0,
        -height / 2.0,
        0.0,
        height / 2.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
    )
}
