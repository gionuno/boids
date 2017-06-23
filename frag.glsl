#version 130

uniform vec3 col;

in vec3 f_x;
in vec2 f_t;

out vec4 o_c;

void main()
{
    o_c.rgb = col;
    o_c.a =1.0;
}

