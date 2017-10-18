/**
 * @file      main.cpp
 * @brief     Main file for CUDA rasterizer. Handles CUDA-GL interop for display.
 * @authors   Skeleton code: Yining Karl Li, Kai Ninomiya, Shuai Shao (Shrek)
 * @date      2012-2016
 * @copyright University of Pennsylvania
 */

#include "main.hpp"

#define STB_IMAGE_IMPLEMENTATION
#define TINYGLTF_LOADER_IMPLEMENTATION
#include <util/tiny_gltf_loader.h>

//-------------------------------
//-------------MAIN--------------
//-------------------------------

int main(int argc, char **argv) 
{
    if (argc != 2) 
	{
        cout << "Usage: [gltf file]. Press Enter to exit" << endl;
		getchar();
        return 0;
    }

	tinygltf::Scene scene;
	tinygltf::TinyGLTFLoader loader;
	std::string err;
	std::string input_filename(argv[1]);
	std::string ext = getFilePathExtension(input_filename);

	bool ret = false;
	if (ext.compare("glb") == 0) 
	{
		// assume binary glTF.
		ret = loader.LoadBinaryFromFile(&scene, &err, input_filename.c_str());
	} 
	else 
	{
		// assume ascii glTF.
		ret = loader.LoadASCIIFromFile(&scene, &err, input_filename.c_str());
	}

	if (!err.empty()) 
	{
		printf("Err: %s\n", err.c_str());
	}

	if (!ret) 
	{
		printf("Failed to parse glTF\n");
		return -1;
	}


    frame = 0;
    seconds = time(NULL);
    fpstracker = 0;

    // Launch CUDA/GL
    if (init(scene)) 
	{
        // GLFW main loop
        mainLoop();
    }

    return 0;
}

void mainLoop() 
{
    while (!glfwWindowShouldClose(window)) 
	{
        glfwPollEvents();
        runCuda();

        time_t seconds2 = time (NULL);

        if (seconds2 - seconds >= 1) 
		{
            fps = fpstracker / (seconds2 - seconds);
            fpstracker = 0;
            seconds = seconds2;
        }

        string title = "CIS565 Rasterizer | " + utilityCore::convertIntToString((int)fps) + " FPS";
        glfwSetWindowTitle(window, title.c_str());

        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
        glBindTexture(GL_TEXTURE_2D, displayImage);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
        glClear(GL_COLOR_BUFFER_BIT);

        // VAO, shader program, and texture already bound
        glDrawElements(GL_TRIANGLES, 6,  GL_UNSIGNED_SHORT, 0);
        glfwSwapBuffers(window);
    }
    glfwDestroyWindow(window);
    glfwTerminate();
}

//-------------------------------
//---------RUNTIME STUFF---------
//-------------------------------
float scale = 1.0f;
float x_trans = 0.0f, y_trans = 0.0f, z_trans = -30.0f;
float x_angle = 0.0f, y_angle = 0.0f;
void runCuda() 
{
    // Map OpenGL buffer object for writing from CUDA on a single GPU
    // No data is moved (Win & Linux). When mapped to CUDA, OpenGL should not use this buffer
    dptr = NULL;

	glm::mat4 P = glm::frustum<float>(-scale * ((float)width) / ((float)height),
									  scale * ((float)width / (float)height),
									  -scale, scale, 1.0, 1000.0);

	glm::mat4 V = glm::mat4(1.0f);

	glm::mat4 M = glm::translate(glm::vec3(x_trans, y_trans, z_trans))
				  * glm::rotate(x_angle, glm::vec3(1.0f, 0.0f, 0.0f))
				  * glm::rotate(y_angle, glm::vec3(0.0f, 1.0f, 0.0f));

	glm::mat3 MV_normal = glm::transpose(glm::inverse(glm::mat3(V) * glm::mat3(M)));
	glm::mat4 MV = V * M;
	glm::mat4 MVP = P * MV;

    cudaGLMapBufferObject((void **)&dptr, pbo);
	rasterize(dptr, MVP, MV, MV_normal);
    cudaGLUnmapBufferObject(pbo);

    frame++;
    fpstracker++;
}

//-------------------------------
//----------SETUP STUFF----------
//-------------------------------

bool init(const tinygltf::Scene & scene) 
{
    glfwSetErrorCallback(errorCallback);

    if (!glfwInit()) 
	{
        return false;
    }

    width = 800;
    height = 800;
    window = glfwCreateWindow(width, height, "CIS 565 Pathtracer", NULL, NULL);
    if (!window) 
	{
        glfwTerminate();
        return false;
    }
    glfwMakeContextCurrent(window);
    glfwSetKeyCallback(window, keyCallback);

    // Set up GL context
    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) 
	{
        return false;
    }

    // Initialize other stuff
    initVAO();
    initTextures();
    initCuda();
	initPBO();

	// Mouse Control Callbacks
	glfwSetMouseButtonCallback(window, mouseButtonCallback);
	glfwSetCursorPosCallback(window, mouseMotionCallback);
	glfwSetScrollCallback(window, mouseWheelCallback);

	{
		std::map<std::string, std::vector<std::string> >::const_iterator it(
			scene.scenes.begin());
		std::map<std::string, std::vector<std::string> >::const_iterator itEnd(
			scene.scenes.end());

		for (; it != itEnd; it++) 
		{
			for (size_t i = 0; i < it->second.size(); i++) 
			{
				std::cout << it->second[i] << ((i != (it->second.size() - 1)) ? ", " : "");
			}
			std::cout << " ] " << std::endl;
		}
	}

	rasterizeSetBuffers(scene);

    GLuint passthroughProgram;
    passthroughProgram = initShader();

    glUseProgram(passthroughProgram);
    glActiveTexture(GL_TEXTURE0);

    return true;
}

void initPBO() 
{
    // set up vertex data parameter
    int num_texels = width * height;
    int num_values = num_texels * 4;
    int size_tex_data = sizeof(GLubyte) * num_values;

    // Generate a buffer ID called a PBO (Pixel Buffer Object)
    glGenBuffers(1, &pbo);

    // Make this the current UNPACK buffer (OpenGL is state-based)
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);

    // Allocate data for the buffer. 4-channel 8-bit image
    glBufferData(GL_PIXEL_UNPACK_BUFFER, size_tex_data, NULL, GL_DYNAMIC_COPY);
    cudaGLRegisterBufferObject(pbo);
}

void initCuda() 
{
    // Use device with highest Gflops/s
    cudaGLSetGLDevice(0);
    rasterizeInit(width, height);

    // Clean up on program exit
    atexit(cleanupCuda);
}

void initTextures() 
{
    glGenTextures(1, &displayImage);
    glBindTexture(GL_TEXTURE_2D, displayImage);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_BGRA,
                  GL_UNSIGNED_BYTE, NULL);
}

void initVAO(void) 
{
    GLfloat vertices[] = {
        -1.0f, -1.0f,
        1.0f, -1.0f,
        1.0f,  1.0f,
        -1.0f,  1.0f,
    };

    GLfloat texcoords[] = {
        1.0f, 1.0f,
        0.0f, 1.0f,
        0.0f, 0.0f,
        1.0f, 0.0f
    };

    GLushort indices[] = { 0, 1, 3, 3, 1, 2 };

    GLuint vertexBufferObjID[3];
    glGenBuffers(3, vertexBufferObjID);

    glBindBuffer(GL_ARRAY_BUFFER, vertexBufferObjID[0]);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
    glVertexAttribPointer((GLuint)positionLocation, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(positionLocation);

    glBindBuffer(GL_ARRAY_BUFFER, vertexBufferObjID[1]);
    glBufferData(GL_ARRAY_BUFFER, sizeof(texcoords), texcoords, GL_STATIC_DRAW);
    glVertexAttribPointer((GLuint)texcoordsLocation, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(texcoordsLocation);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vertexBufferObjID[2]);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);
}


GLuint initShader() 
{
    const char *attribLocations[] = { "Position", "Tex" };
    GLuint program = glslUtility::createDefaultProgram(attribLocations, 2);
    GLint location;

    glUseProgram(program);
    if ((location = glGetUniformLocation(program, "u_image")) != -1) 
	{
        glUniform1i(location, 0);
    }

    return program;
}

//-------------------------------
//---------CLEANUP STUFF---------
//-------------------------------
void cleanupCuda() 
{
    if (pbo) 
	{
        deletePBO(&pbo);
    }
    if (displayImage) 
	{
        deleteTexture(&displayImage);
    }
}

void deletePBO(GLuint *pbo) 
{
    if (pbo) 
	{
        // unregister this buffer object with CUDA
        cudaGLUnregisterBufferObject(*pbo);

        glBindBuffer(GL_ARRAY_BUFFER, *pbo);
        glDeleteBuffers(1, pbo);

        *pbo = (GLuint)NULL;
    }
}

void deleteTexture(GLuint *tex) 
{
    glDeleteTextures(1, tex);
    *tex = (GLuint)NULL;
}

void shut_down(int return_code) 
{
    rasterizeFree();
    cudaDeviceReset();
#ifdef __APPLE__
    glfwTerminate();
#endif
    exit(return_code);
}

//------------------------------
//-------GLFW CALLBACKS---------
//------------------------------
void errorCallback(int error, const char *description) 
{
    fputs(description, stderr);
}

void keyCallback(GLFWwindow *window, int key, int scancode, int action, int mods) 
{
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) 
	{
        glfwSetWindowShouldClose(window, GL_TRUE);
    }
}

//----------------------------
//----- util -----------------
//----------------------------
static std::string getFilePathExtension(const std::string &FileName) 
{
	if (FileName.find_last_of(".") != std::string::npos)
		return FileName.substr(FileName.find_last_of(".") + 1);
	return "";
}



//-----------------------------
//---- Mouse control ----------
//-----------------------------

enum ControlState { NONE = 0, ROTATE, TRANSLATE };
ControlState mouseState = NONE;
void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods)
{
	if (action == GLFW_PRESS)
	{
		if (button == GLFW_MOUSE_BUTTON_LEFT)
		{
			mouseState = ROTATE;
		}
		else if (button == GLFW_MOUSE_BUTTON_RIGHT)
		{
			mouseState = TRANSLATE;
		}

	}
	else if (action == GLFW_RELEASE)
	{
		mouseState = NONE;
	}
}

double lastx = (double)width / 2;
double lasty = (double)height / 2;
void mouseMotionCallback(GLFWwindow* window, double xpos, double ypos)
{
	const double s_r = 0.01;
	const double s_t = 0.01;

	double diffx = xpos - lastx;
	double diffy = ypos - lasty;
	lastx = xpos;
	lasty = ypos;

	if (mouseState == ROTATE)
	{
		//rotate
		x_angle += (float)s_r * diffy;
		y_angle += (float)s_r * diffx;
	}
	else if (mouseState == TRANSLATE)
	{
		//translate
		x_trans += (float)(s_t * diffx);
		y_trans += (float)(-s_t * diffy);
	}
}

void mouseWheelCallback(GLFWwindow* window, double xoffset, double yoffset)
{
	const double s = 1.0;	// sensitivity
	z_trans += (float)(s * yoffset);
}
