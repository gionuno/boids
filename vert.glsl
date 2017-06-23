#version 130

in vec3 v_x;
in vec2 v_t;

uniform vec2 pos;
uniform float ang;

out vec3 f_x;
out vec2 f_t;

void main()
{
	f_x = vec3(0.02*(cos(ang)*v_x.x-sin(ang)*v_x.y)+pos.x,0.02*(sin(ang)*v_x.x+cos(ang)*v_x.y)+pos.y,0.0);
	f_t = v_t;
	gl_Position = vec4(f_x,1.);
}
