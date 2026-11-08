#version 450
layout(location = 0) in vec3 a_position;
layout(location = 1) in vec3 a_color;
layout(location = 0) out vec3 v_color;
layout(push_constant) uniform Push { mat4 mvp; } pc;
void main()
{
    gl_Position = pc.mvp * vec4(a_position, 1.0);
    v_color = a_color;
}
