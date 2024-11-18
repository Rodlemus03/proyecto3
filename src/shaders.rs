use noise::Simplex;
use noise::NoiseFn;
use nalgebra_glm::{Vec3, Vec4, Mat3, dot, mat4_to_mat3};
use crate::vertex::Vertex;
use crate::Uniforms;
use crate::fragment::Fragment;
use crate::color::Color;
use std::f32::consts::PI;
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
use fastnoise_lite::NoiseType;
use crate::Uniforms_Simplex;
use crate::FastNoiseLite;

pub fn vertex_shader(vertex: &Vertex, uniforms: &Uniforms) -> Vertex {
    let position = Vec4::new(
        vertex.position.x,
        vertex.position.y,
        vertex.position.z,
        1.0
    );

    let transformed = uniforms.projection_matrix * uniforms.view_matrix * uniforms.MM * position;

    let w = transformed.w;
    let transformed_position = Vec4::new(
        transformed.x / w,
        transformed.y / w,
        transformed.z / w,
        1.0
    );

    let screen_position = uniforms.viewport_matrix * transformed_position;

    let model_mat3 = mat4_to_mat3(&uniforms.MM);
    let normal_matrix = model_mat3.transpose().try_inverse().unwrap_or(Mat3::identity());

    let transformed_normal = normal_matrix * vertex.normal;

    Vertex {
        position: vertex.position,
        normal: vertex.normal,
        tex_coords: vertex.tex_coords,
        color: vertex.color,
        transformed_position: Vec3::new(screen_position.x, screen_position.y, screen_position.z),
        transformed_normal: transformed_normal
    }
}

pub fn vertex_shader_simplex(vertex: &Vertex, uniforms: &Uniforms_Simplex) -> Vertex {
  let position = Vec4::new(
      vertex.position.x,
      vertex.position.y,
      vertex.position.z,
      1.0
  );

  let transformed = uniforms.projection_matrix * uniforms.view_matrix * uniforms.MM * position;

  let w = transformed.w;
  let transformed_position = Vec4::new(
      transformed.x / w,
      transformed.y / w,
      transformed.z / w,
      1.0
  );

  let screen_position = uniforms.viewport_matrix * transformed_position;

  let model_mat3 = mat4_to_mat3(&uniforms.MM);
  let normal_matrix = model_mat3.transpose().try_inverse().unwrap_or(Mat3::identity());

  let transformed_normal = normal_matrix * vertex.normal;

  Vertex {
      position: vertex.position,
      normal: vertex.normal,
      tex_coords: vertex.tex_coords,
      color: vertex.color,
      transformed_position: Vec3::new(screen_position.x, screen_position.y, screen_position.z),
      transformed_normal: transformed_normal
  }
}

pub fn fragment_shader(fragment: &Fragment, uniforms: &Uniforms) -> Color {
    planeta_gaseoso(fragment, uniforms)
    // dalmata_shader(fragment, uniforms)
    // cloud_shader(fragment, uniforms)
    // cellular_shader(fragment, uniforms)
    // lava_shader(fragment, uniforms)
}

fn ruido_perlin(x: f32, y: f32) -> f32 {
  (x.sin() * y.cos()) * 0.5
}

fn ruido_con_fractal_strata(simplex: &Simplex, x: f64, y: f64, octaves: usize, persistence: f64) -> f64 {
  let mut amplitude = 10.0;
  let mut frequency = 20.0;
  let mut ruido = 20.0;
  let mut max_value = 0.0;

  for _ in 0..octaves {
      let value = simplex.get([x * frequency, y * frequency]);

      ruido += (value * amplitude).abs();
      max_value += amplitude;
      amplitude *= persistence;
      frequency *= 2.0;
  }

  ruido / max_value
}

pub fn shader_luna(fragment: &Fragment, uniforms: &Uniforms_Simplex) -> Color {
  let zoom = 0.5;
  let ox = 50.0;
  let oy = 50.0;
  let x = fragment.vertex_position.x * zoom + ox;
  let y = fragment.vertex_position.y * zoom + oy;

  let ruido = ruido_con_fractal_strata(&uniforms.noise, x as f64, y as f64, 5, 0.5);

  let color_base = Color::new(128, 128, 128);
  let factor = (ruido + 1.0) as f32 / 2.0; 
  let color_final = color_base * factor;

  color_final * fragment.intensity
}

pub fn planeta_gaseoso(fragment: &Fragment, uniforms: &Uniforms) -> Color {
  let color3 = Color::new(245, 245, 220);    
  let color1 = Color::new(255, 255, 255);    
  let color2 = Color::new(173, 216, 230);    
  let color4 = Color::new(25, 25, 112);      
  let color5 = Color::new(112, 128, 144);    

  let x = fragment.vertex_position.x;
  let y = fragment.vertex_position.y;

  let tiempo = (uniforms.time as f32) * 0.01;  

  let frecuencia = 10.0;
  let distancia = (x * x + y * y).sqrt();

  let ruido = ruido_perlin(x * 0.5 + tiempo, y * 0.5);  

  let angulo = tiempo * 0.5;  


  let patron1 = ((distancia + ruido) * 7.0 * frecuencia + (y + ruido) * 5.0 + angulo).sin() * 0.5 + 0.5;
  let patron2 = ((distancia + ruido) * 5.0 * frecuencia - (y + ruido) * 8.0 + PI / 3.0 + angulo).sin() * 0.5 + 0.5;
  let patron3 = ((distancia + ruido) * 6.0 * frecuencia + (x + ruido) * 4.0 + 2.0 * PI / 3.0 + angulo).sin() * 0.5 + 0.5;

  let mut color_final = color1.lerp(&color2, patron1);
  color_final = color_final.lerp(&color3, patron2);
  color_final = color_final.lerp(&color4, patron3);
  color_final = color_final.lerp(&color5, patron2);

  color_final * fragment.intensity
}

pub fn black_and_azul(fragment: &Fragment, uniforms: &Uniforms) -> Color {
    let seed = uniforms.time as f32 * fragment.vertex_position.y * fragment.vertex_position.x;
  
    let mut rng = StdRng::seed_from_u64(seed.abs() as u64);
  
    let random_number = rng.gen_range(0..=100);
  
    let black_or_azul = if random_number < 50 {
      Color::new(0, 0, 0)
    } else {
      Color::new(255, 255, 255)
    };
  
    black_or_azul * fragment.intensity
}
  
pub fn dalmata_shader(fragment: &Fragment, uniforms: &Uniforms) -> Color {
    let zoom = 100.0;
    let ox = 0.0;
    let oy = 0.0;
    let x = fragment.vertex_position.x;
    let y = fragment.vertex_position.y;
  
    let ruido = uniforms.noise.get_noise_2d(
      (x + ox) * zoom,
      (y + oy) * zoom,
    );
  
    let spot_threshold = 0.5;
    let spot_color = Color::new(255, 255, 255); 
    let base_color = Color::new(0, 0, 0); 
  
    let noise_color = if ruido < spot_threshold {
      spot_color
    } else {
      base_color
    };
  
    noise_color * fragment.intensity
}
  
pub fn cloud_shader(fragment: &Fragment, uniforms: &Uniforms) -> Color {
    let zoom = 100.0;  
    let ox = 100.0; 
    let oy = 100.0;
    let x = fragment.vertex_position.x;
    let y = fragment.vertex_position.y;
    let t = uniforms.time as f32 * 0.5;
  
    let ruido = uniforms.noise.get_noise_2d(x * zoom + ox + t, y * zoom + oy);
  
   
    let cloud_threshold = 0.5; 
    let cloud_color = Color::new(255, 255, 255); 
    let sky_color = Color::new(30, 97, 145); 
  
    
    let noise_color = if ruido > cloud_threshold {
      cloud_color
    } else {
      sky_color
    };
  
    noise_color * fragment.intensity
}
  
pub fn cellular_shader(fragment: &Fragment, uniforms: &Uniforms) -> Color {
  let zoom = 300.0;  
  let ox = 100.0;    
  let oy = 100.0;    
  let x = fragment.vertex_position.x;
  let y = fragment.vertex_position.y;

  let cell_ruido = uniforms.noise.get_noise_2d(x * zoom + ox, y * zoom + oy).abs();

  let color_piedra = Color::new(180, 120, 60);  
  let color_mas_oscuro = Color::new(110, 50, 10);  

  let factor = (cell_ruido * cell_ruido) * 10.0;  
  let color_final = color_mas_oscuro.lerp(&color_piedra, factor.clamp(0.0, 1.0));

  color_final * fragment.intensity
}

pub fn lava_shader(fragment: &Fragment, uniforms: &Uniforms) -> Color {
  let color1 = Color::new(243, 83, 23);  
  let color2 = Color::new(205, 64, 19);  
  let color3 = Color::new(165, 28, 5);  


  let position = Vec3::new(
      fragment.vertex_position.x,
      fragment.vertex_position.y,
      fragment.depth,
  );

  let base_frecuencia = 1.0;
  let pulsate_amplitude = 1.5;
  let t = uniforms.time as f32 * 0.01;

  let pulsate = (t * base_frecuencia).sin() * pulsate_amplitude;

  let zoom = 100.0;
  let ruido1 = uniforms.noise.get_noise_3d(
      position.x * zoom,
      position.y * zoom,
      (position.z + pulsate) * zoom,
  );
  let ruido2 = uniforms.noise.get_noise_3d(
      (position.x + 1000.0) * zoom,
      (position.y + 1000.0) * zoom,
      (position.z + 1000.0 + pulsate) * zoom,
  );
  let ruido = (ruido1 + ruido2) * 0.5;

  let val_normalizado = ruido.clamp(0.0, 1.0);

  let color_intermediate = color1.lerp(&color2, val_normalizado);
  let final_color = color_intermediate.lerp(&color3, val_normalizado);

  final_color * fragment.intensity
}

pub fn anillos_shader(fragment: &Fragment, uniforms: &Uniforms) -> Color {
  let color1 = Color::new(220, 200, 180);  
  let color2 = Color::new(150, 100, 70);   
  let color3 = Color::new(50, 30, 20);     

  let position = Vec3::new(
      fragment.vertex_position.x,
      fragment.vertex_position.y,
      fragment.depth,
  );

  let base_frecuencia = 0.5;
  let pulsate_amplitude = 1.0;
  let t = uniforms.time as f32 * 0.01;

  let pulsate = (t * base_frecuencia).sin() * pulsate_amplitude;

  let zoom = 100.0;
  let ruido1 = uniforms.noise.get_noise_3d(
      position.x * zoom,
      position.y * zoom,
      (position.z + pulsate) * zoom,
  );
  let ruido2 = uniforms.noise.get_noise_3d(
      (position.x + 1000.0) * zoom,
      (position.y + 1000.0) * zoom,
      (position.z + 1000.0 + pulsate) * zoom,
  );
  let ruido = (ruido1 + ruido2) * 0.5;

  let val_normalizado = ruido.clamp(0.0, 1.0);

  let color_intermediate = color1.lerp(&color2, val_normalizado);
  let final_color = color_intermediate.lerp(&color3, val_normalizado);

  final_color * fragment.intensity
}


pub fn shader_puntas(fragment: &Fragment, uniforms: &Uniforms) -> Color {
  let green_color = Color::new(0, 255, 0);  
  let blue_color = Color::new(0, 0, 255);  

  let position = Vec3::new(
      fragment.vertex_position.x,
      fragment.vertex_position.y,
      fragment.depth,
  );

  let zoom = 200.0; 
  let vel = 0.1; 
  let tiempo = uniforms.time as f32 * vel;

  let ruido = uniforms.noise.get_noise_3d(
      position.x * zoom + tiempo, 
      position.y * zoom,        
      position.z * zoom + tiempo,
  );

  let val_normalizado = (ruido + 1.0) / 2.0;

  let mixed_color = green_color.lerp(&blue_color, val_normalizado);

  mixed_color * fragment.intensity
}

pub fn shader_grupos(fragment: &Fragment, uniforms: &Uniforms) -> Color {
  let rojo = Color::new(47, 79, 79);  
  let amarillo = Color::new(0, 128, 128);  
  let azul = Color::new(255, 255, 255); 

  let position = Vec3::new(
      fragment.vertex_position.x,
      fragment.vertex_position.y,
      fragment.depth,
  );

  let zoom = 200.0;
  let vel = 0.1; 
  let tiempo = uniforms.time as f32 * vel;


  let ruido = uniforms.noise.get_noise_3d(
      position.x * zoom + tiempo, 
      position.y * zoom,        
      position.z * zoom + tiempo, 
  );


  let val_normalizado = (ruido + 1.0) / 2.0;

  let mixed_color = rojo.lerp(&amarillo, val_normalizado);
  let final_color = mixed_color.lerp(&azul, val_normalizado);

  final_color * fragment.intensity
}

pub fn bacteria_shader(fragment: &Fragment, uniforms: &Uniforms) -> Color {
  let rojo = Color::new(138, 43, 226);  
  let amarillo = Color::new(64, 224, 208); 
  let azul = Color::new(255, 105, 180); 

  let position = Vec3::new(
      fragment.vertex_position.x,
      fragment.vertex_position.y,
      fragment.depth,
  );

  let zoom = 200.0;
  let vel = 0.1;
  let tiempo = uniforms.time as f32 * vel;

  let ruido = uniforms.noise.get_noise_3d(
      position.x * zoom + tiempo,
      position.y * zoom,       
      position.z * zoom + tiempo, 
  );

  let val_normalizado = (ruido + 1.0) / 2.0;

  let mixed_color = rojo.lerp(&amarillo, val_normalizado);
  let final_color = mixed_color.lerp(&azul, val_normalizado);

  final_color * fragment.intensity
}

pub fn shader_variado(fragment: &Fragment, uniforms: &Uniforms) -> Color {
  let coral = Color::new(255, 127, 80);  
  let verde_agua = Color::new(144, 238, 144);  
  let crema = Color::new(255, 253, 208);  
  let position = Vec3::new(
      fragment.vertex_position.x,
      fragment.vertex_position.y,
      fragment.depth,
  );


  let zoom = 200.0; 
  let vel = 2.0; 
  let tiempo = uniforms.time as f32 * vel;

  let ruido = uniforms.noise.get_noise_3d(
      position.x * zoom + tiempo, 
      position.y * zoom,        
      position.z * zoom + tiempo, 
  );

  let val_normalizado = (ruido + 1.0) / 2.0;

  let mixed_color = coral.lerp(&verde_agua, val_normalizado);
  let final_color = mixed_color.lerp(&crema, val_normalizado);

  final_color * fragment.intensity
}

pub fn camo_shader(fragment: &Fragment, uniforms: &Uniforms) -> Color {
  let rojo = Color::new(255, 69, 0);  
  let amarillo = Color::new(255, 223, 0);  
  let azul = Color::new(30, 144, 255);  

  let position = Vec3::new(
      fragment.vertex_position.x,
      fragment.vertex_position.y,
      fragment.depth,
  );

  let zoom = 20.0; 
  let vel = 0.05; 
  let tiempo = uniforms.time as f32 * vel;

  let ruido = uniforms.noise.get_noise_3d(
      position.x * zoom + tiempo, 
      position.y * zoom,        
      position.z * zoom + tiempo, 
  );

  let val_normalizado = (ruido + 1.0) / 2.0;

  let mixed_color = rojo.lerp(&amarillo, val_normalizado);
  let final_color = mixed_color.lerp(&azul, val_normalizado);

  final_color * fragment.intensity
}

fn ruido_fractal(noise: &FastNoiseLite, x: f32, y: f32, octaves: u32, lacunarity: f32, gain: f32) -> f32 {
  let mut total = 10.0;
  let mut frequency = 20.0;
  let mut amplitude = 30.0;
  let mut max_value = 0.0; 

  for _ in 0..octaves {
      total += noise.get_noise_2d(x * frequency, y * frequency) * amplitude;
      max_value += amplitude;

      amplitude *= gain;
      frequency *= lacunarity;
  }

  total / max_value 
}

pub fn shader_agua(fragment: &Fragment, uniforms: &Uniforms) -> Color {
  let agua_1 = Color::new(0, 105, 148);  
  let agua_2 = Color::new(0, 191, 255);  
  let color_h = Color::new(173, 216, 230);  

  let position = fragment.vertex_position;
  let t = uniforms.time as f32 * 0.02;  

  let ruido = ruido_fractal(&uniforms.noise, position.x + t, position.y + t, 5, 2.0, 0.5);

  let olas = (1.0 + ruido) * 0.5; 
  let base_color = agua_1.lerp(&agua_2, olas);
  let final_color = base_color.lerp(&color_h, ruido.abs());

  final_color * fragment.intensity
}

pub fn shader_agujero_negro(fragment: &Fragment, uniforms: &Uniforms) -> Color {
  let centro_blanco = Color::new(0, 0, 0);        
  let rojo_fuego = Color::new(255, 69, 0);         
  let naranja_fuego = Color::new(255, 140, 0);     
  let negro_externo = Color::new(0, 0, 0);         

  let position = Vec3::new(
      fragment.vertex_position.x,
      fragment.vertex_position.y,
      fragment.depth,
  );

  let distance_from_center = position.x.hypot(position.y);

  let radio_interno = 0.3; 
  let radio_externo = 0.7; 
  let dist = ((distance_from_center - radio_interno) / (radio_externo - radio_interno)).clamp(0.0, 1.0);

  let zoom = 200.0;  
  let vel = 0.001;   
  let tiempo = uniforms.time as f32 * vel;

  let ruido = uniforms.noise.get_noise_3d(
      position.x * zoom + tiempo, 
      position.y * zoom + tiempo, 
      (position.z + tiempo) * zoom, 
  );

  let normalized_noise = (ruido + 1.0) / 2.0;

  let color_fuego = rojo_fuego.lerp(&naranja_fuego, dist);
  let color_con_sonido = color_fuego.lerp(&naranja_fuego, normalized_noise);

  let final_color = if distance_from_center < radio_interno {
      centro_blanco 
  } else if distance_from_center < radio_externo {
      color_con_sonido
  } else {
      negro_externo 
  };


  final_color * fragment.intensity
}

