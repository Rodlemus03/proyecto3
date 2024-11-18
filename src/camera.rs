use nalgebra_glm::{Vec3, rotate_vec3, normalize, cross};

pub struct Camera {
    pub eye: Vec3,
    pub center: Vec3,
    pub up: Vec3,
    pub has_changed: bool,
}

impl Camera {
    pub fn new(eye: Vec3, center: Vec3, up: Vec3) -> Self {
        Camera {
            eye,
            center,
            up,
            has_changed: true,
        }
    }

    pub fn get_forward(&self) -> Vec3 {
        normalize(&(self.center - self.eye))
    }

    pub fn get_right(&self) -> Vec3 {
        cross(&self.get_forward(), &self.up)
    }

    pub fn get_up(&self) -> Vec3 {
        self.up
    }

    pub fn orbit(&mut self, delta_yaw: f32, delta_pitch: f32) {
        let radius_vector = self.eye - self.center;
        let radius = radius_vector.magnitude();

        let current_yaw = radius_vector.z.atan2(radius_vector.x);
        let radius_xz = (radius_vector.x * radius_vector.x + radius_vector.z * radius_vector.z).sqrt();
        let current_pitch = (-radius_vector.y).atan2(radius_xz);

        let new_yaw = (current_yaw + delta_yaw) % (2.0 * std::f32::consts::PI);
        let new_pitch = (current_pitch + delta_pitch).clamp(-std::f32::consts::PI / 2.0 + 0.1, std::f32::consts::PI / 2.0 - 0.1);

        let new_eye = self.center + Vec3::new(
            radius * new_yaw.cos() * new_pitch.cos(),
            -radius * new_pitch.sin(),
            radius * new_yaw.sin() * new_pitch.cos(),
        );

        self.eye = new_eye;
        self.has_changed = true;
    }

    pub fn zoom(&mut self, delta: f32) {
        let direction = normalize(&(self.center - self.eye));
        self.eye += direction * delta;
        self.has_changed = true;
    }

    pub fn move_center(&mut self, direction: Vec3) {
        self.eye += direction;
        self.center += direction;
        self.has_changed = true;
    }

    pub fn rotate_y(&mut self, angle: f32) {
        let forward = self.get_forward();
        let new_forward = rotate_vec3(&forward, angle, &self.up); 
        self.center = self.eye + new_forward;
        self.has_changed = true;
    }


    pub fn rotate_x(&mut self, angle: f32) {
        let right = self.get_right();
        let forward = self.get_forward();
        let new_forward = rotate_vec3(&forward, angle, &right); 
        self.center = self.eye + new_forward;
        self.has_changed = true;
    }

    pub fn check_if_changed(&mut self) -> bool {
        if self.has_changed {
            self.has_changed = false;
            true
        } else {
            false
        }
    }

    pub fn base_change(&self, direction: &Vec3) -> Vec3 {
        let forward = self.get_forward();
        let right = self.get_right();
        let up = self.get_up();

        Vec3::new(
            right.dot(direction),
            up.dot(direction),
            forward.dot(direction),
        )
    }
}
