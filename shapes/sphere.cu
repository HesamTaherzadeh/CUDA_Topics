#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <cmath>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

// Sphere properties
const int SPHERE_RES = 50;
const float SPHERE_RADIUS = 1.0f;

struct Vertex {
    float x, y, z;
};

__host__ __device__ Vertex operator+(const Vertex &a, const Vertex &b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

__host__ __device__ Vertex operator-(const Vertex &a, const Vertex &b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

__host__ __device__ Vertex operator*(const Vertex &a, float b) {
    return {a.x * b, a.y * b, a.z * b};
}

__host__ __device__ Vertex normalize(const Vertex &v) {
    float norm = sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
    return {v.x / norm, v.y / norm, v.z / norm};
}

__global__ void generateSphere(Vertex *vertices, int res, float radius) {
    int thetaIndex = blockIdx.x * blockDim.x + threadIdx.x;
    int phiIndex = blockIdx.y * blockDim.y + threadIdx.y;

    if (thetaIndex >= res || phiIndex >= res) return;

    float theta = thetaIndex * 2.0f * M_PI / res;
    float phi = phiIndex * M_PI / res;

    float x = radius * sinf(phi) * cosf(theta);
    float y = radius * sinf(phi) * sinf(theta);
    float z = radius * cosf(phi);

    int index = phiIndex * res + thetaIndex;
    vertices[index] = {x, y, z};
}

// Camera control variables
float cameraDistance = 5.0f;
float cameraAngleX = 0.0f;
float cameraAngleY = 0.0f;
bool mousePressed = false;
double lastMouseX, lastMouseY;

const char* vertexShaderSource = R"(
#version 330 core
layout(location = 0) in vec3 aPos;
uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
void main()
{
    gl_Position = projection * view * model * vec4(aPos, 1.0);
}
)";

const char* fragmentShaderSource = R"(
#version 330 core
out vec4 FragColor;
void main()
{
    FragColor = vec4(1.0, 1.0, 1.0, 1.0);
}
)";

void display(GLFWwindow *window, GLuint vbo, GLuint vao, int numVertices, GLuint shaderProgram) {
    while (!glfwWindowShouldClose(window)) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // Set up the viewport and projection matrix
        int width, height;
        glfwGetFramebufferSize(window, &width, &height);
        glViewport(0, 0, width, height);

        glm::mat4 projection = glm::perspective(glm::radians(45.0f), (float)width / (float)height, 0.1f, 100.0f);
        glm::mat4 view = glm::lookAt(glm::vec3(cameraDistance * sin(cameraAngleX) * cos(cameraAngleY),
                                               cameraDistance * sin(cameraAngleY),
                                               cameraDistance * cos(cameraAngleX) * cos(cameraAngleY)),
                                     glm::vec3(0.0f, 0.0f, 0.0f),
                                     glm::vec3(0.0f, 1.0f, 0.0f));
        glm::mat4 model = glm::mat4(1.0f);

        glUseProgram(shaderProgram);

        GLint modelLoc = glGetUniformLocation(shaderProgram, "model");
        GLint viewLoc = glGetUniformLocation(shaderProgram, "view");
        GLint projectionLoc = glGetUniformLocation(shaderProgram, "projection");

        glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(model));
        glUniformMatrix4fv(viewLoc, 1, GL_FALSE, glm::value_ptr(view));
        glUniformMatrix4fv(projectionLoc, 1, GL_FALSE, glm::value_ptr(projection));

        glBindVertexArray(vao);
        glDrawArrays(GL_POINTS, 0, numVertices);  // Use GL_POINTS for simplicity
        glBindVertexArray(0);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }
}

void keyCallback(GLFWwindow *window, int key, int scancode, int action, int mods) {
    if (action == GLFW_PRESS || action == GLFW_REPEAT) {
        if (key == GLFW_KEY_UP) {
            cameraDistance -= 0.1f; // Zoom in
            if (cameraDistance < 1.0f) cameraDistance = 1.0f; // Prevent zooming too close
            std::cout << "Zoomed in. Current distance: " << cameraDistance << std::endl;
        } else if (key == GLFW_KEY_DOWN) {
            cameraDistance += 0.1f; // Zoom out
            std::cout << "Zoomed out. Current distance: " << cameraDistance << std::endl;
        }
    }
}

void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods) {
    if (button == GLFW_MOUSE_BUTTON_LEFT) {
        if (action == GLFW_PRESS) {
            mousePressed = true;
            glfwGetCursorPos(window, &lastMouseX, &lastMouseY);
        } else if (action == GLFW_RELEASE) {
            mousePressed = false;
        }
    }
}

void cursorPositionCallback(GLFWwindow* window, double xpos, double ypos) {
    if (mousePressed) {
        double dx = xpos - lastMouseX;
        double dy = ypos - lastMouseY;
        cameraAngleX += 0.01f * dx;
        cameraAngleY += 0.01f * dy;
        lastMouseX = xpos;
        lastMouseY = ypos;
    }
}

GLuint compileShader(GLenum type, const char* source) {
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, NULL);
    glCompileShader(shader);

    int success;
    char infoLog[512];
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(shader, 512, NULL, infoLog);
        std::cerr << "ERROR::SHADER::COMPILATION_FAILED\n" << infoLog << std::endl;
    }
    return shader;
}

GLuint createShaderProgram() {
    GLuint vertexShader = compileShader(GL_VERTEX_SHADER, vertexShaderSource);
    GLuint fragmentShader = compileShader(GL_FRAGMENT_SHADER, fragmentShaderSource);

    GLuint shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);

    int success;
    char infoLog[512];
    glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
    if (!success) {
        glGetProgramInfoLog(shaderProgram, 512, NULL, infoLog);
        std::cerr << "ERROR::PROGRAM::LINKING_FAILED\n" << infoLog << std::endl;
    }

    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    return shaderProgram;
}

int main() {
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return -1;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow *window = glfwCreateWindow(800, 600, "CUDA Sphere", NULL, NULL);
    if (!window) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);
    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) {
        std::cerr << "Failed to initialize GLEW" << std::endl;
        return -1;
    }

    glEnable(GL_DEPTH_TEST);

    int numVertices = SPHERE_RES * SPHERE_RES;
    Vertex *d_vertices;
    cudaMalloc(&d_vertices, numVertices * sizeof(Vertex));

    dim3 blockSize(16, 16);
    dim3 gridSize((SPHERE_RES + blockSize.x - 1) / blockSize.x, (SPHERE_RES + blockSize.y - 1) / blockSize.y);
    generateSphere<<<gridSize, blockSize>>>(d_vertices, SPHERE_RES, SPHERE_RADIUS);
    cudaDeviceSynchronize();

    Vertex *h_vertices = new Vertex[numVertices];
    cudaMemcpy(h_vertices, d_vertices, numVertices * sizeof(Vertex), cudaMemcpyDeviceToHost);

    for (int i = 0; i < std::min(numVertices, 10); ++i) {
        std::cout << "Vertex " << i << ": (" << h_vertices[i].x << ", " << h_vertices[i].y << ", " << h_vertices[i].z << ")" << std::endl;
    }

    GLuint vbo, vao;
    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);

    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, numVertices * sizeof(Vertex), h_vertices, GL_STATIC_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)0);
    glEnableVertexAttribArray(0);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    GLuint shaderProgram = createShaderProgram();

    glfwSetKeyCallback(window, keyCallback);
    glfwSetMouseButtonCallback(window, mouseButtonCallback);
    glfwSetCursorPosCallback(window, cursorPositionCallback);

    display(window, vbo, vao, numVertices, shaderProgram);

    cudaFree(d_vertices);
    delete[] h_vertices;

    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}
