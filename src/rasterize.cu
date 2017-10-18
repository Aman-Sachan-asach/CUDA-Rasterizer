/**
 * @file      rasterize.cu
 * @brief     CUDA-accelerated rasterization pipeline.
 * @authors   Skeleton code: Yining Karl Li, Kai Ninomiya, Shuai Shao (Shrek)
 * @date      2012-2016
 * @copyright University of Pennsylvania & STUDENT
 */

#include <cmath>
#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/random.h>
#include <thrust/execution_policy.h>
#include <util/checkCUDAError.h>
#include <util/tiny_gltf_loader.h>
#include "rasterizeTools.h"
#include "rasterize.h"
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <chrono>

static const int DEPTHSCALE = INT_MAX;
static const int numTilesX = 32;
static const int numTilesY = 32;
static const int maxNumTiles = (numTilesX + 1)*(numTilesY + 1);

//--------------------
//Toggle-able OPTIONS
//--------------------
// only use tilebased or scanline not both
//Tile Based Rasterization -- only does triangular rasterization
#define TILEBASED 0
	#define DISPLAY_TILES 0

//Scanline Rasterization
#define SCANLINE 1
	#define RASTERIZE_TRIANGLES 0;
	#define RASTERIZE_LINES 1;
	#define RASTERIZE_POINTS 0;

//Shading stuff handled in the render function
#define DISPLAY_DEPTH 0
#define DISPLAY_NORMAL 0
#define DISPLAY_ABSNORMAL 0
#define FRAG_SHADING_LAMBERT 1

//texture stuff
#define TEXTURE_MAPPING 1
#define BILINEAR_FILTERING 1

//Depth Testing and Culling
#define DEPTH_TEST 1
#define BACKFACE_CULLING 0

namespace 
{
	typedef unsigned short VertexIndex;
	typedef glm::vec3 VertexAttributePosition;
	typedef glm::vec3 VertexAttributeNormal;
	typedef glm::vec2 VertexAttributeTexcoord;
	typedef unsigned char TextureData;

	typedef unsigned char BufferByte;

	enum PrimitiveType
	{
		Point = 1,
		Line = 2,
		Triangle = 3
	};

	struct VertexOut 
	{
		glm::vec4 vPos;
		glm::vec3 vEyePos;	// eye space position used for shading
		glm::vec3 vNor;	// eye space normal used for shading, cuz normal will go wrong after perspective transformation
		glm::vec3 vColor;
		glm::vec2 texcoord0;
		TextureData* dev_diffuseTex = NULL;
		int texWidth, texHeight;
	};

	struct Tile {
		int triIndices[1000]; //indices of the triangles that each pixel in the tile has to check
							 //limit to 500 triangles in a tile
	};
	
	struct Primitive 
	{
		PrimitiveType primitiveType = Triangle;	// C++ 11 init
		VertexOut v[3];
		bool tileBuckets[maxNumTiles];
		bool cull;
	};

	struct Fragment 
	{
		glm::vec3 fColor;
		glm::vec3 fEyePos;	// eye space position used for shading
		glm::vec3 fNor;
		float depth;
		VertexAttributeTexcoord texcoord0;
		TextureData* dev_diffuseTex;
	};

	struct PrimitiveDevBufPointers 
	{
		int primitiveMode;	//from tinygltfloader macro
		PrimitiveType primitiveType;
		int numPrimitives;
		int numIndices;
		int numVertices;

		// Vertex In, const after loaded
		VertexIndex* dev_indices;
		VertexAttributePosition* dev_position;
		VertexAttributeNormal* dev_normal;
		VertexAttributeTexcoord* dev_texcoord0;

		// Materials, add more attributes when needed
		TextureData* dev_diffuseTex;
		int diffuseTexWidth;
		int diffuseTexHeight;
		// TextureData* dev_specularTex;
		// TextureData* dev_normalTex;
		// ...

		// Vertex Out, vertex used for rasterization, this is changing every frame
		VertexOut* dev_verticesOut;

		// TODO: add more attributes when needed
	};
}

static std::map<std::string, std::vector<PrimitiveDevBufPointers>> mesh2PrimitivesMap;

static int width = 0;
static int height = 0;

static int totalNumPrimitives = 0;
static int numActivePrimitives = 0;

static Primitive *dev_primitives = NULL;
static Fragment *dev_fragmentBuffer = NULL;
static glm::vec3 *dev_framebuffer = NULL;

static Tile *dev_tiles = NULL;
static int* dev_tileTriCount = NULL; //how many triangles have actually filled the list
static int* dev_tilemutex = NULL;

static int * dev_depth = NULL; //depth buffer
static int * dev_mutex = NULL; //mutex buffer for depth

//------------------------------------------------
//-------------------Timer------------------------
using time_point_t = std::chrono::high_resolution_clock::time_point;
time_point_t timeStartCpu;
time_point_t timeEndCpu;
float prevElapsedTime = 0.0f;
//------------------------------------------------

// Kernel that writes the image to the OpenGL PBO directly.
__global__ 
void sendImageToPBO(uchar4 *pbo, int w, int h, glm::vec3 *image) 
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    int index = x + (y * w);

    if (x < w && y < h) 
	{
        glm::vec3 fcolor;
        fcolor.x = glm::clamp(image[index].x, 0.0f, 1.0f) * 255.0;
        fcolor.y = glm::clamp(image[index].y, 0.0f, 1.0f) * 255.0;
        fcolor.z = glm::clamp(image[index].z, 0.0f, 1.0f) * 255.0;
        // Each thread writes one pixel location in the texture (textel)
        pbo[index].w = 0;
        pbo[index].x = fcolor.x;
        pbo[index].y = fcolor.y;
        pbo[index].z = fcolor.z;
    }
}

__host__ __device__ 
glm::vec3 LambertFragShader(glm::vec3 pos, glm::vec3 color, glm::vec3 normal)
{
	glm::vec3 lightPosition = glm::vec3(1.0f);
	glm::vec3 finalColor = color*glm::dot(normal, glm::normalize(lightPosition - pos));
	return finalColor;
}

//Writes fragment colors to the framebuffer
__global__ 
void render(int w, int h, Fragment *fragmentBuffer, glm::vec3 *framebuffer) 
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    int index = x + (y * w);

    if (x < w && y < h) 
	{
		#if SCANLINE
			#if RASTERIZE_TRIANGLES
				#if DISPLAY_DEPTH
						framebuffer[index] = glm::vec3(fragmentBuffer[index].depth);
				#elif DISPLAY_NORMAL
						framebuffer[index] = fragmentBuffer[index].fNor;
				#elif DISPLAY_ABSNORMAL
						framebuffer[index] = glm::abs(fragmentBuffer[index].fNor);
				#elif FRAG_SHADING_LAMBERT
						framebuffer[index] = LambertFragShader(fragmentBuffer[index].fEyePos,
															   fragmentBuffer[index].fColor,
															   fragmentBuffer[index].fNor);
				#else
						framebuffer[index] = fragmentBuffer[index].fColor + 0.15f;
				#endif
			#endif
			#if RASTERIZE_LINES
					framebuffer[index] = fragmentBuffer[index].fColor + 0.15f;
			#endif
			#if RASTERIZE_POINTS
					framebuffer[index] = fragmentBuffer[index].fColor + 0.15f;
			#endif
		#endif
		#if TILEBASED
			#if DISPLAY_TILES
					framebuffer[index] = fragmentBuffer[index].fColor;
			#elif DISPLAY_DEPTH
								framebuffer[index] = glm::vec3(fragmentBuffer[index].depth);
			#elif DISPLAY_NORMAL
								framebuffer[index] = fragmentBuffer[index].fNor;
			#elif DISPLAY_ABSNORMAL
								framebuffer[index] = glm::abs(fragmentBuffer[index].fNor);
			#elif FRAG_SHADING_LAMBERT
								framebuffer[index] = LambertFragShader(fragmentBuffer[index].fEyePos,
									fragmentBuffer[index].fColor,
									fragmentBuffer[index].fNor);
			#else
					framebuffer[index] = fragmentBuffer[index].fColor + 0.15f;
			#endif
		#endif
    }
}

//Called once at the beginning of the program to allocate memory.
void rasterizeInit(int w, int h) 
{
    width = w;
    height = h;
	cudaFree(dev_fragmentBuffer);
	cudaMalloc(&dev_fragmentBuffer, width * height * sizeof(Fragment));
	cudaMemset(dev_fragmentBuffer, 0, width * height * sizeof(Fragment));
    cudaFree(dev_framebuffer);
    cudaMalloc(&dev_framebuffer,   width * height * sizeof(glm::vec3));
    cudaMemset(dev_framebuffer, 0, width * height * sizeof(glm::vec3));
    
	cudaFree(dev_tiles);
	cudaMalloc(&dev_tiles, maxNumTiles * sizeof(Tile));
	cudaMemset(dev_tiles, 0, maxNumTiles * sizeof(Tile));

	cudaFree(dev_tileTriCount);
	cudaMalloc(&dev_tileTriCount, maxNumTiles * sizeof(int));

	cudaFree(dev_tilemutex);
	cudaMalloc(&dev_tilemutex, maxNumTiles * sizeof(int));

	cudaFree(dev_depth);
	cudaMalloc(&dev_depth, width * height * sizeof(int));

	cudaFree(dev_mutex);
	cudaMalloc(&dev_mutex, width * height * sizeof(int));
	
	checkCUDAError("rasterizeInit");
}

__global__ 
void initDepth(int w, int h, int * depth)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < w && y < h)
	{
		int index = x + (y * w);
		depth[index] = INT_MAX;
	}
}

__global__
void initCullValue(int numPrimitives, Primitive * dev_primitive)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index < numPrimitives)
	{
		dev_primitive[index].cull = false;
	}
}

//kern function with support for stride to sometimes replace cudaMemcpy
//One thread is responsible for copying one component
__global__ 
void _deviceBufferCopy(int N, BufferByte* dev_dst, const BufferByte* dev_src, int n, int byteStride, int byteOffset, int componentTypeByteSize) 
{
	// Attribute (vec3 position)
	// component (3 * float)
	// byte (4 * byte)

	// id of component
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (i < N) 
	{
		int count = i / n;
		int offset = i - count * n;	// which component of the attribute

		for (int j = 0; j < componentTypeByteSize; j++) 
		{
			dev_dst[count * componentTypeByteSize * n 
				+ offset * componentTypeByteSize 
				+ j]

				= 

			dev_src[byteOffset 
				+ count * (byteStride == 0 ? componentTypeByteSize * n : byteStride) 
				+ offset * componentTypeByteSize 
				+ j];
		}
	}
}

__global__
void _nodeMatrixTransform( int numVertices,
						   VertexAttributePosition* position,
						   VertexAttributeNormal* normal,
						   glm::mat4 MV, glm::mat3 MV_normal) 
{
	// vertex id
	int vid = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (vid < numVertices) 
	{
		position[vid] = glm::vec3(MV * glm::vec4(position[vid], 1.0f));
		normal[vid] = glm::normalize(MV_normal * normal[vid]);
	}
}

glm::mat4 getMatrixFromNodeMatrixVector(const tinygltf::Node & n) 
{
	glm::mat4 curMatrix(1.0);

	const std::vector<double> &m = n.matrix;
	if (m.size() > 0) 
	{
		// matrix, copy it
		for (int i = 0; i < 4; i++) 
		{
			for (int j = 0; j < 4; j++) 
			{
				curMatrix[i][j] = (float)m.at(4 * i + j);
			}
		}
	} 
	else 
	{
		// no matrix, use rotation, scale, translation
		if (n.translation.size() > 0) 
		{
			curMatrix[3][0] = n.translation[0];
			curMatrix[3][1] = n.translation[1];
			curMatrix[3][2] = n.translation[2];
		}

		if (n.rotation.size() > 0) 
		{
			glm::mat4 R;
			glm::quat q;
			q[0] = n.rotation[0];
			q[1] = n.rotation[1];
			q[2] = n.rotation[2];

			R = glm::mat4_cast(q);
			curMatrix = curMatrix * R;
		}

		if (n.scale.size() > 0) 
		{
			curMatrix = curMatrix * glm::scale(glm::vec3(n.scale[0], n.scale[1], n.scale[2]));
		}
	}

	return curMatrix;
}

void traverseNode (	std::map<std::string, glm::mat4> & n2m,
					const tinygltf::Scene & scene,
					const std::string & nodeString,
					const glm::mat4 & parentMatrix ) 
{
	const tinygltf::Node & n = scene.nodes.at(nodeString);
	glm::mat4 M = parentMatrix * getMatrixFromNodeMatrixVector(n);
	n2m.insert(std::pair<std::string, glm::mat4>(nodeString, M));

	auto it = n.children.begin();
	auto itEnd = n.children.end();

	for (; it != itEnd; ++it) 
	{
		traverseNode(n2m, scene, *it, M);
	}
}

void rasterizeSetBuffers(const tinygltf::Scene & scene) 
{
	totalNumPrimitives = 0;

	std::map<std::string, BufferByte*> bufferViewDevPointers;

	// 1. copy all `bufferViews` to device memory
	{
		std::map<std::string, tinygltf::BufferView>::const_iterator it(
			scene.bufferViews.begin());
		std::map<std::string, tinygltf::BufferView>::const_iterator itEnd(
			scene.bufferViews.end());

		for (; it != itEnd; it++) 
		{
			const std::string key = it->first;
			const tinygltf::BufferView &bufferView = it->second;
			if (bufferView.target == 0) 
			{
				continue; // Unsupported bufferView.
			}

			const tinygltf::Buffer &buffer = scene.buffers.at(bufferView.buffer);

			BufferByte* dev_bufferView;
			cudaMalloc(&dev_bufferView, bufferView.byteLength);
			cudaMemcpy(dev_bufferView, &buffer.data.front() + bufferView.byteOffset, bufferView.byteLength, cudaMemcpyHostToDevice);

			checkCUDAError("Set BufferView Device Mem");

			bufferViewDevPointers.insert(std::make_pair(key, dev_bufferView));
		}
	}

	// 2. for each mesh: 
	//		for each primitive: 
	//			build device buffer of indices, materail, and each attributes
	//			and store these pointers in a map
	{
		std::map<std::string, glm::mat4> nodeString2Matrix;
		auto rootNodeNamesList = scene.scenes.at(scene.defaultScene);

		{
			auto it = rootNodeNamesList.begin();
			auto itEnd = rootNodeNamesList.end();
			for (; it != itEnd; ++it) {
				traverseNode(nodeString2Matrix, scene, *it, glm::mat4(1.0f));
			}
		}

		// parse through node to access mesh

		auto itNode = nodeString2Matrix.begin();
		auto itEndNode = nodeString2Matrix.end();
		for (; itNode != itEndNode; ++itNode) {

			const tinygltf::Node & N = scene.nodes.at(itNode->first);
			const glm::mat4 & matrix = itNode->second;
			const glm::mat3 & matrixNormal = glm::transpose(glm::inverse(glm::mat3(matrix)));

			auto itMeshName = N.meshes.begin();
			auto itEndMeshName = N.meshes.end();

			for (; itMeshName != itEndMeshName; ++itMeshName) {

				const tinygltf::Mesh & mesh = scene.meshes.at(*itMeshName);

				auto res = mesh2PrimitivesMap.insert(std::pair<std::string, std::vector<PrimitiveDevBufPointers>>(mesh.name, std::vector<PrimitiveDevBufPointers>()));
				std::vector<PrimitiveDevBufPointers> & primitiveVector = (res.first)->second;

				// for each primitive
				for (size_t i = 0; i < mesh.primitives.size(); i++) {
					const tinygltf::Primitive &primitive = mesh.primitives[i];

					if (primitive.indices.empty())
						return;

					// TODO: add new attributes for your PrimitiveDevBufPointers when you add new attributes
					VertexIndex* dev_indices = NULL;
					VertexAttributePosition* dev_position = NULL;
					VertexAttributeNormal* dev_normal = NULL;
					VertexAttributeTexcoord* dev_texcoord0 = NULL;

					// ----------Indices-------------

					const tinygltf::Accessor &indexAccessor = scene.accessors.at(primitive.indices);
					const tinygltf::BufferView &bufferView = scene.bufferViews.at(indexAccessor.bufferView);
					BufferByte* dev_bufferView = bufferViewDevPointers.at(indexAccessor.bufferView);

					// assume type is SCALAR for indices
					int n = 1;
					int numIndices = indexAccessor.count;
					int componentTypeByteSize = sizeof(VertexIndex);
					int byteLength = numIndices * n * componentTypeByteSize;

					dim3 numThreadsPerBlock(128);
					dim3 numBlocks((numIndices + numThreadsPerBlock.x - 1) / numThreadsPerBlock.x);
					cudaMalloc(&dev_indices, byteLength);
					_deviceBufferCopy <<<numBlocks, numThreadsPerBlock>>> ( numIndices,
																			(BufferByte*)dev_indices,
																			dev_bufferView,
																			n,
																			indexAccessor.byteStride,
																			indexAccessor.byteOffset,
																			componentTypeByteSize );


					checkCUDAError("Set Index Buffer");


					// ---------Primitive Info-------

					// Warning: LINE_STRIP is not supported in tinygltfloader
					int numPrimitives;
					PrimitiveType primitiveType;
					switch (primitive.mode) {
					case TINYGLTF_MODE_TRIANGLES:
						primitiveType = PrimitiveType::Triangle;
						numPrimitives = numIndices / 3;
						break;
					case TINYGLTF_MODE_TRIANGLE_STRIP:
						primitiveType = PrimitiveType::Triangle;
						numPrimitives = numIndices - 2;
						break;
					case TINYGLTF_MODE_TRIANGLE_FAN:
						primitiveType = PrimitiveType::Triangle;
						numPrimitives = numIndices - 2;
						break;
					case TINYGLTF_MODE_LINE:
						primitiveType = PrimitiveType::Line;
						numPrimitives = numIndices / 2;
						break;
					case TINYGLTF_MODE_LINE_LOOP:
						primitiveType = PrimitiveType::Line;
						numPrimitives = numIndices + 1;
						break;
					case TINYGLTF_MODE_POINTS:
						primitiveType = PrimitiveType::Point;
						numPrimitives = numIndices;
						break;
					default:
						// output error
						break;
					};


					// ----------Attributes-------------

					auto it(primitive.attributes.begin());
					auto itEnd(primitive.attributes.end());

					int numVertices = 0;
					// for each attribute
					for (; it != itEnd; it++) 
					{
						const tinygltf::Accessor &accessor = scene.accessors.at(it->second);
						const tinygltf::BufferView &bufferView = scene.bufferViews.at(accessor.bufferView);

						int n = 1;
						if (accessor.type == TINYGLTF_TYPE_SCALAR) {
							n = 1;
						}
						else if (accessor.type == TINYGLTF_TYPE_VEC2) {
							n = 2;
						}
						else if (accessor.type == TINYGLTF_TYPE_VEC3) {
							n = 3;
						}
						else if (accessor.type == TINYGLTF_TYPE_VEC4) {
							n = 4;
						}

						BufferByte * dev_bufferView = bufferViewDevPointers.at(accessor.bufferView);
						BufferByte ** dev_attribute = NULL;

						numVertices = accessor.count;
						int componentTypeByteSize;

						// Note: since the type of our attribute array (dev_position) is static (float32)
						// We assume the glTF model attribute type are 5126(FLOAT) here

						if (it->first.compare("POSITION") == 0) {
							componentTypeByteSize = sizeof(VertexAttributePosition) / n;
							dev_attribute = (BufferByte**)&dev_position;
						}
						else if (it->first.compare("NORMAL") == 0) {
							componentTypeByteSize = sizeof(VertexAttributeNormal) / n;
							dev_attribute = (BufferByte**)&dev_normal;
						}
						else if (it->first.compare("TEXCOORD_0") == 0) {
							componentTypeByteSize = sizeof(VertexAttributeTexcoord) / n;
							dev_attribute = (BufferByte**)&dev_texcoord0;
						}

						std::cout << accessor.bufferView << "  -  " << it->second << "  -  " << it->first << '\n';

						dim3 numThreadsPerBlock(128);
						dim3 numBlocks((n * numVertices + numThreadsPerBlock.x - 1) / numThreadsPerBlock.x);
						int byteLength = numVertices * n * componentTypeByteSize;
						cudaMalloc(dev_attribute, byteLength);

						_deviceBufferCopy <<<numBlocks, numThreadsPerBlock>>> ( n * numVertices,
																				*dev_attribute,
																				dev_bufferView,
																				n,
																				accessor.byteStride,
																				accessor.byteOffset,
																				componentTypeByteSize);

						std::string msg = "Set Attribute Buffer: " + it->first;
						checkCUDAError(msg.c_str());
					}

					// malloc for VertexOut
					VertexOut* dev_vertexOut;
					cudaMalloc(&dev_vertexOut, numVertices * sizeof(VertexOut));
					checkCUDAError("Malloc VertexOut Buffer");

					// ----------Materials-------------

					// You can only worry about this part once you started to 
					// implement textures for your rasterizer
					TextureData* dev_diffuseTex = NULL;
					int diffuseTexWidth = 0;
					int diffuseTexHeight = 0;
					if (!primitive.material.empty()) {
						const tinygltf::Material &mat = scene.materials.at(primitive.material);
						printf("material.name = %s\n", mat.name.c_str());

						if (mat.values.find("diffuse") != mat.values.end()) {
							std::string diffuseTexName = mat.values.at("diffuse").string_value;
							if (scene.textures.find(diffuseTexName) != scene.textures.end()) {
								const tinygltf::Texture &tex = scene.textures.at(diffuseTexName);
								if (scene.images.find(tex.source) != scene.images.end()) {
									const tinygltf::Image &image = scene.images.at(tex.source);

									size_t s = image.image.size() * sizeof(TextureData);
									cudaMalloc(&dev_diffuseTex, s);
									cudaMemcpy(dev_diffuseTex, &image.image.at(0), s, cudaMemcpyHostToDevice);
									
									diffuseTexWidth = image.width;
									diffuseTexHeight = image.height;

									checkCUDAError("Set Texture Image data");
								}
							}
						}

						// TODO: write your code for other materails
						// You may have to take a look at tinygltfloader
						// You can also use the above code loading diffuse material as a start point 
					}


					// ---------Node hierarchy transform--------
					cudaDeviceSynchronize();
					
					dim3 numBlocksNodeTransform((numVertices + numThreadsPerBlock.x - 1) / numThreadsPerBlock.x);
					_nodeMatrixTransform << <numBlocksNodeTransform, numThreadsPerBlock >> > (
						numVertices,
						dev_position,
						dev_normal,
						matrix,
						matrixNormal);

					checkCUDAError("Node hierarchy transformation");

					// at the end of the for loop of primitive
					// push dev pointers to map
					primitiveVector.push_back(PrimitiveDevBufPointers{
						primitive.mode,
						primitiveType,
						numPrimitives,
						numIndices,
						numVertices,

						dev_indices,
						dev_position,
						dev_normal,
						dev_texcoord0,

						dev_diffuseTex,
						diffuseTexWidth,
						diffuseTexHeight,

						dev_vertexOut	//VertexOut
					});

					totalNumPrimitives += numPrimitives;

				} // for each primitive

			} // for each mesh

		} // for each node

	}
	

	// 3. Malloc for dev_primitives and dev_tiles(do it here instead of 
	//memory management on a per frame basis)
	{
		cudaMalloc(&dev_primitives, totalNumPrimitives * sizeof(Primitive));
	}
	

	// Finally, cudaFree raw dev_bufferViews
	{

		std::map<std::string, BufferByte*>::const_iterator it(bufferViewDevPointers.begin());
		std::map<std::string, BufferByte*>::const_iterator itEnd(bufferViewDevPointers.end());
			
			//bufferViewDevPointers

		for (; it != itEnd; it++) {
			cudaFree(it->second);
		}

		checkCUDAError("Free BufferView Device Mem");
	}
}

__global__ 
void _vertexTransformAndAssembly( int numVertices, PrimitiveDevBufPointers primitive, 
								  glm::mat4 MVP, glm::mat4 MV, glm::mat3 MV_normal, 
								  int width, int height ) 
{
	// vertex id
	int vid = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (vid < numVertices) 
	{
		//---------------------------------------------------
		//-------------- Vertex Transformation --------------
		//---------------------------------------------------
		// Multiply the MVP matrix for each vertex position, this will transform everything into clipping space
		// Then divide the pos by its w element to transform into NDC space
		// Finally transform x and y to viewport space
		glm::vec4 vPos = glm::vec4(primitive.dev_position[vid], 1.0f);
		glm::vec4 eyePos = MV*vPos;
		vPos = MVP*vPos; //now things are in clip space
		vPos /= vPos.w; //now things are in NDC space
		vPos.x = (vPos.x + 1.0f)*float(width)*0.5f;
		vPos.y = (1.0f - vPos.y)*float(height)*0.5f; //now in pixel space or window coordinates
		
		vPos.z = (vPos.z+1.0f)*0.5f; // to convert z from a 1 to -1 range to a 0 to 1 range

		glm::vec3 vNor = primitive.dev_normal[vid];
		vNor = glm::normalize(MV_normal*vNor);
		//---------------------------------------------------
		//-------------- Vertex assembly --------------------
		//---------------------------------------------------
		// Assemble all attribute arrays into the primitive array
		primitive.dev_verticesOut[vid].vPos = vPos;
		primitive.dev_verticesOut[vid].vNor = vNor;
		primitive.dev_verticesOut[vid].vEyePos = glm::vec3(eyePos);
		primitive.dev_verticesOut[vid].vColor = glm::vec3(0,1,0);

		// Texture Mapping
		if (primitive.dev_diffuseTex == NULL) 
		{
			primitive.dev_verticesOut[vid].dev_diffuseTex = NULL;
		}
		else 
		{
			primitive.dev_verticesOut[vid].texcoord0 = primitive.dev_texcoord0[vid];
			primitive.dev_verticesOut[vid].dev_diffuseTex = primitive.dev_diffuseTex;
			primitive.dev_verticesOut[vid].texWidth = primitive.diffuseTexWidth;
			primitive.dev_verticesOut[vid].texHeight = primitive.diffuseTexHeight;
		}
	}
}

static int curPrimitiveBeginId = 0;

__global__ 
void _primitiveAssembly(int numIndices, int curPrimitiveBeginId, 
						Primitive* dev_primitives, PrimitiveDevBufPointers primitive) 
{
	// index id
	int iid = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (iid < numIndices) 
	{
		// This is primitive assembly for triangles
		int pid;	// id for cur primitives vector
		if (primitive.primitiveMode == TINYGLTF_MODE_TRIANGLES) 
		{
			pid = iid / (int)primitive.primitiveType;
			dev_primitives[pid + curPrimitiveBeginId].v[iid % (int)primitive.primitiveType]
				= primitive.dev_verticesOut[primitive.dev_indices[iid]];
		}

		// TODO: other primitive types (point, line)
	}
}

__host__ __device__ static
glm::vec3 getTextureColorAt(const TextureData* texture, const int& textureWidth, int& u, int& v)
{
	int flatIndex = (u + v * textureWidth) * 3;
	float r = (float)texture[flatIndex] / 255.0f; //flatIndex * 3 --> because 3 color channels
	float g = (float)texture[flatIndex + 1] / 255.0f;
	float b = (float)texture[flatIndex + 2] / 255.0f;
	return glm::vec3(r, g, b);
}

__host__ __device__ static
glm::vec3 getBilinearFilteredColor(const TextureData* tex,
								   const int &texWidth, const int &texHeight,
								   const float &u, const float &v)
{
	//references: 
	//https://en.wikipedia.org/wiki/Bilinear_filtering
	//https://www.scratchapixel.com/lessons/mathematics-physics-for-computer-graphics/interpolation/bilinear-filtering
	float x = u * (float)texWidth;
	float y = v * (float)texHeight;
	float floorX = glm::floor(x);
	float floorY = glm::floor(y);
	float deltaX = x - floorX;
	float deltaY = y - floorY;

	//get the square for which we will perform bilinear interpolation
	int xPos = (int)floorX;
	int yPos = (int)floorY;
	int xPlusOne = glm::clamp(xPos + 1, 0, texWidth - 1);
	int yPlusOne = glm::clamp(yPos + 1, 0, texHeight - 1);

	//get 4 color values
	glm::vec3 c00 = getTextureColorAt(tex, texWidth, xPos, yPos);
	glm::vec3 c10 = getTextureColorAt(tex, texWidth, xPlusOne, yPos);
	glm::vec3 c01 = getTextureColorAt(tex, texWidth, xPos, yPlusOne);
	glm::vec3 c11 = getTextureColorAt(tex, texWidth, xPlusOne, yPlusOne);

	//bilinear interpolation between the above 4 colors
	glm::vec3 c20 = glm::mix(c00, c10, deltaX);
	glm::vec3 c21 = glm::mix(c01, c11, deltaX);
	return glm::mix(c20, c21, deltaY);
}

__host__ __device__ 
void modifyFragment(Primitive* dev_primitives, Fragment* dev_fragments, 
					int* dev_depthBuffer, float& z,
					glm::vec3 tri[3], glm::vec3 baryCoords,
					int& index, int& fragIndex)
{
	glm::vec3 v0eyePos = dev_primitives[index].v[0].vEyePos;
	glm::vec3 v1eyePos = dev_primitives[index].v[1].vEyePos;
	glm::vec3 v2eyePos = dev_primitives[index].v[2].vEyePos;

	//for perspective correct interpolation you need the z values
	float z1 = v0eyePos.z;
	float z2 = v1eyePos.z;
	float z3 = v2eyePos.z;
	float perpectiveCorrectZ = 1.0f/(baryCoords.x / v0eyePos.z + 
									 baryCoords.y / v1eyePos.z + 
									 baryCoords.z / v2eyePos.z );

	glm::vec3 v0color = dev_primitives[index].v[0].vColor;
	glm::vec3 v1color = dev_primitives[index].v[1].vColor;
	glm::vec3 v2color = dev_primitives[index].v[2].vColor;

	glm::vec3 v0Nor = dev_primitives[index].v[0].vNor;
	glm::vec3 v1Nor = dev_primitives[index].v[1].vNor;
	glm::vec3 v2Nor = dev_primitives[index].v[2].vNor;

	glm::vec2 v0UV = dev_primitives[index].v[0].texcoord0;
	glm::vec2 v1UV = dev_primitives[index].v[1].texcoord0;
	glm::vec2 v2UV = dev_primitives[index].v[2].texcoord0;

	TextureData* triangleDiffuseTex = dev_primitives[index].v[0].dev_diffuseTex;

	//if testing Depth coloration
	dev_fragments[fragIndex].dev_diffuseTex = triangleDiffuseTex;
	dev_fragments[fragIndex].depth = dev_depthBuffer[fragIndex]/float(DEPTHSCALE);
	dev_fragments[fragIndex].fNor = perpectiveCorrectZ*((v0Nor / z1)*baryCoords.x +
														(v1Nor / z2)*baryCoords.y + 
														(v2Nor / z3)*baryCoords.z );
	dev_fragments[fragIndex].texcoord0 = perpectiveCorrectZ*((v0UV / z1)*baryCoords.x +
															 (v0UV / z2)*baryCoords.y +
															 (v0UV / z3)*baryCoords.z);

	if (TEXTURE_MAPPING && dev_fragments[fragIndex].dev_diffuseTex != NULL)
	{
#if BILINEAR_FILTERING
		dev_fragments[fragIndex].fColor = getBilinearFilteredColor(dev_fragments[fragIndex].dev_diffuseTex,
																   dev_primitives[index].v[0].texWidth,
																   dev_primitives[index].v[0].texHeight,
																   dev_fragments[fragIndex].texcoord0[0],
																   dev_fragments[fragIndex].texcoord0[1]);
#else
		int u = dev_fragments[fragIndex].texcoord0[0] * dev_primitives[index].v[0].texWidth;
		int v = dev_fragments[fragIndex].texcoord0[1] * dev_primitives[index].v[0].texHeight;
		dev_fragments[fragIndex].fColor = getTextureColorAt(dev_fragments[fragIndex].dev_diffuseTex,
															dev_primitives[index].v[0].texWidth, u, v);
#endif
	}
	else
	{
		dev_fragments[fragIndex].fColor = perpectiveCorrectZ*((v0color / z1)*baryCoords.x +
															  (v1color / z2)*baryCoords.y +
															  (v2color / z3)*baryCoords.z);
	}

	//to make the normals follow convention:
	//z is positive coming out of the screen
	//x is positive to the right
	//y is positive going up
	dev_fragments[fragIndex].fNor.x *= -1.0f;

	//clamp color and normals values
	dev_fragments[fragIndex].fNor.x = glm::clamp(dev_fragments[fragIndex].fNor.x, 0.0f, 1.0f);
	dev_fragments[fragIndex].fNor.y = glm::clamp(dev_fragments[fragIndex].fNor.y, 0.0f, 1.0f);
	dev_fragments[fragIndex].fNor.z = glm::clamp(dev_fragments[fragIndex].fNor.z, 0.0f, 1.0f);

	dev_fragments[fragIndex].fColor.x = glm::clamp(dev_fragments[fragIndex].fColor.x, 0.0f, 1.0f);
	dev_fragments[fragIndex].fColor.y = glm::clamp(dev_fragments[fragIndex].fColor.y, 0.0f, 1.0f);
	dev_fragments[fragIndex].fColor.z = glm::clamp(dev_fragments[fragIndex].fColor.z, 0.0f, 1.0f);
}

__device__ 
void DepthTest(Primitive* dev_primitives, Fragment* dev_fragments,
			   int* dev_depthBuffer, int * dev_mutex, float& z,
			   glm::vec3 tri[3], glm::vec3 baryCoords,
			   int& index, int& fragIndex)
{
	//multiplying z value by a large static int because atomicCAS is only defined for ints
	//and atomicCAS is needed to handle race conditions along with the mutex lock
	int scaledZ = z*DEPTHSCALE;

	bool isSet;
	do
	{
		isSet = (atomicCAS(&dev_mutex[fragIndex], 0, 1) == 0);
		if (isSet)
		{
			// Critical section goes here.
			// if it is afterward, a deadlock will occur.
			if (scaledZ < dev_depthBuffer[fragIndex])
			{
				dev_depthBuffer[fragIndex] = scaledZ;
				modifyFragment(dev_primitives, dev_fragments, dev_depthBuffer, z,
									tri, baryCoords, index, fragIndex);
			}

			dev_mutex[fragIndex] = 0;
		}
	} while (!isSet);
}

struct predicate_PrimitiveCulling
{
	__host__ __device__ bool operator()(const Primitive &x)
	{
		return (x.cull);
	}
};

__global__
void identifyBackFaces(const int numActivePrimitives, Primitive* prims, const glm::vec3 camForward)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	
	if (index < numActivePrimitives)
	{
		//check if the normal of the triangle face the camera or not
		glm::vec3 p1 = glm::vec3(prims[index].v[0].vPos);
		glm::vec3 p2 = glm::vec3(prims[index].v[1].vPos);
		glm::vec3 p3 = glm::vec3(prims[index].v[2].vPos);

		glm::vec3 triangleNormal = glm::cross(p1 - p2, p2 - p3);
		float dot = glm::dot(triangleNormal, camForward);

		if (dot < 0.0f)
		{
			//cull this triangle
			prims[index].cull = true;
		}
	}
}

void BackFaceCulling(int& numActivePrimitives, Primitive* dev_primitives, glm::vec3& camForward)
{
	dim3 numThreadsPerBlock(128);
	dim3 blockSize1d((numActivePrimitives - 1) / numThreadsPerBlock.x + 1);

	initCullValue <<<blockSize1d, numThreadsPerBlock>>> (numActivePrimitives, dev_primitives);

	//identify and mark the triangles to be culled
	identifyBackFaces <<<blockSize1d, numThreadsPerBlock>>> (numActivePrimitives, dev_primitives, camForward);
	checkCUDAError("face identification failed");

	//Stream Compact your array of dev_primitives to cull out primitives that cant be seen or lit in a scene
	//thrust::partition returns a pointer to the element in the array where the partition occurs 
	Primitive* partition_point = thrust::partition(thrust::device, 
												   dev_primitives, 
												   dev_primitives + numActivePrimitives, 
												   predicate_PrimitiveCulling());
	checkCUDAError("partitioning and streamcompaction failed");
	numActivePrimitives = int(partition_point - dev_primitives);
}

__device__
void _rasterizeTriangles(int w, int h, int index, glm::vec3 *tri,
						 Primitive* dev_primitives, Fragment* dev_fragments, 
						 int* dev_depthBuffer, int* dev_mutex)
{
	AABB boundingBox = getAABBForTriangle(tri);

	//clamp BB to be within the window
	int BBminX = glm::min(w-1, glm::max(0, int(boundingBox.min.x)));
	int BBmaxX = glm::max(0, glm::min(w-1, int(boundingBox.max.x)));
	int BBminY = glm::min(h-1, glm::max(0, int(boundingBox.min.y)));
	int BBmaxY = glm::max(0, glm::min(h-1, int(boundingBox.max.y)));

	for (int y = BBminY; y <= BBmaxY; ++y)
	{
		for (int x = BBminX; x <= BBmaxX; ++x)
		{
			glm::vec3 baryCoords = calculateBarycentricCoordinate(tri, glm::vec2(x, y));
			bool isInsideTriangle = isBarycentricCoordInBounds(baryCoords);
			if (isInsideTriangle)
			{
				int fragIndex = x + y*w;
				float z = -getZAtCoordinate(baryCoords, tri);
#if DEPTH_TEST
				DepthTest(dev_primitives, dev_fragments, dev_depthBuffer, dev_mutex,
					z, tri, baryCoords, index, fragIndex);
#else
				modifyFragment(dev_primitives, dev_fragments, dev_depthBuffer, z,
					tri, baryCoords, index, fragIndex);
#endif
			}
		}
	}
}

__device__
void _rasterizeTriangleAsLines(int width, int height, const int *indicies,
								Fragment* dev_fragments, glm::vec3 *tri)
{
	int x1, x2, y1, y2, dx, dy, y, fragIndex;
	for (int index = 0; index < 6; index += 2)
	{
		x1 = tri[indicies[index]].x;    
		y1 = tri[indicies[index]].y;
		x2 = tri[indicies[index + 1]].x;  
		y2 = tri[indicies[index + 1]].y;
		dx = x2 - x1;                   
		dy = y2 - y1;
		for (int x = x1; x <= x2; x++)
		{
			y = y1 + dy * (x - x1) / dx;
			fragIndex = x + y * width;
			if ((x >= 0 && x <= width - 1) && 
				(y >= 0 && y <= height - 1))
			{
				dev_fragments[fragIndex].fColor = glm::vec3(0.0f, 0.0f, 1.0f);
			}				
		}
	}
}

__device__
void _rasterizeTriangleAsPoints(int width, int height, Fragment* dev_fragments, glm::vec3 *tri)
{
	int x, y, fragIndex;
	for (int vertexId = 0; vertexId < 3; ++vertexId)
	{
		x = tri[vertexId].x;
		y = tri[vertexId].y;
		int fragIndex = x + y * width;
		if ((x >= 0 && x <= width - 1) && 
			(y >= 0 && y <= height - 1))
		{
			dev_fragments[fragIndex].fColor = glm::vec3(1.0f, 0.0f, 0.0f);
		}
	}
}

__global__ 
void _rasterizeScanLine(int w, int h, int numTriangles, Primitive* dev_primitives, 
						Fragment* dev_fragments, int* dev_depthBuffer, int* dev_mutex)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index < numTriangles)
	{
		glm::vec3 tri[3];
		tri[0] = glm::vec3(dev_primitives[index].v[0].vPos);
		tri[1] = glm::vec3(dev_primitives[index].v[1].vPos);
		tri[2] = glm::vec3(dev_primitives[index].v[2].vPos);

#if RASTERIZE_TRIANGLES
		_rasterizeTriangles(w, h, index, tri, dev_primitives, dev_fragments, dev_depthBuffer, dev_mutex);
#endif
#if RASTERIZE_LINES
		const int indices[] = { 0,1,1,2,2,0 };
		_rasterizeTriangleAsLines(w, h, indices, dev_fragments, tri);
#endif
#if RASTERIZE_POINTS
		_rasterizeTriangleAsPoints(w, h, dev_fragments, tri);
#endif

	}
}

__global__ 
void RasterizePixels(int pixelXoffset, int pixelYoffset , int numpixelsX, int numpixelsY, 
					 int imageWidth, int tileID, Tile* dev_tiles, 
					 Primitive* dev_primitives, Fragment* dev_fragments, 
					 int * dev_tileTriCount,
					 int* dev_depthBuffer, int* dev_mutex)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int index = (x+pixelXoffset) + ((y+ pixelYoffset) * imageWidth);
		
	if (x < numpixelsX && y < numpixelsY)
	{
		//Each thread loops over the triangles inside the tile
		//Discard tiles (ie kernel launches) that dont have any triangles inside them --> implicitly done by for loop
		for (int i = 0; i < dev_tileTriCount[tileID]; i++)
		{
#if DISPLAY_TILES
			dev_fragments[index].fColor = glm::vec3(1, 0, 0);
			return;
#endif
			int triangleIndex = dev_tiles[tileID].triIndices[i];
			glm::vec3 tri[3];
			tri[0] = glm::vec3(dev_primitives[triangleIndex].v[0].vPos);
			tri[1] = glm::vec3(dev_primitives[triangleIndex].v[1].vPos);
			tri[2] = glm::vec3(dev_primitives[triangleIndex].v[2].vPos);

			int _x = (x + pixelXoffset);
			int _y = (y+ pixelYoffset);

			glm::vec3 baryCoords = calculateBarycentricCoordinate(tri, glm::vec2(_x, _y));
			bool isInsideTriangle = isBarycentricCoordInBounds(baryCoords);
			if (isInsideTriangle)
			{
				int fragIndex = index;
				float z = -getZAtCoordinate(baryCoords, tri);
#if DEPTH_TEST
				DepthTest(dev_primitives, dev_fragments, dev_depthBuffer, dev_mutex,
					z, tri, baryCoords, triangleIndex, fragIndex);
#else
				modifyFragment(dev_primitives, dev_fragments, dev_depthBuffer, z,
					tri, baryCoords, triangleIndex, fragIndex);
#endif
			}
		}	
	}
}

__global__ void bucketPrims_TileMutex(int w, int stride_x, int stride_y,
									int numTriangles, 
									Primitive* dev_primitives,
									Tile* tiles,
									int * tileTriCount, 
									int * dev_tilemutex)
{
	//does the bucketing of primitives into tiles but tries to avoid race conditions by updating a list of bools that 
	//correspond to the tiles the window is divided into. The list of bools exists per primitive.
	//This bool list is later compiled per primitive to get the total number of primitives in a tile
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index < numTriangles)
	{
		glm::vec3 tri[3];
		tri[0] = glm::vec3(dev_primitives[index].v[0].vPos);
		tri[1] = glm::vec3(dev_primitives[index].v[1].vPos);
		tri[2] = glm::vec3(dev_primitives[index].v[2].vPos);
		AABB boundingBox = getAABBForTriangle(tri);

		//if boundingbox of triangle lies inside tile add it to tile triangle list
		int tilesX = (int)glm::ceil(double(w) / double(stride_x));

		int tileidminX = glm::floor(boundingBox.min.x / stride_x);
		int tileidmaxX = glm::ceil(boundingBox.max.x / stride_x);
		int tileidminY = glm::floor(boundingBox.min.y / stride_y);
		int tileidmaxY = glm::ceil(boundingBox.max.y / stride_y);
					
		//use mutex lock
		for (int i = tileidminY; i < tileidmaxY; i++)
		{
			for (int j = tileidminX; j < tileidmaxX; j++)
			{
				int tileID = j + i*(tilesX);
				bool isSet;
				do
				{
					isSet = (atomicCAS(&dev_tilemutex[tileID], 0, 1) == 0);
					if (isSet)
					{
						// Critical section goes here.
						// if it is afterward, a deadlock will occur.
						int t = tileTriCount[tileID];
						tiles[tileID].triIndices[t] = index;
						tileTriCount[tileID] = t + 1;

						dev_tilemutex[tileID] = 0;
					}
				} while (!isSet);

			}
		}
	}
}

__global__ 
void resetTiles(int numTiles, int stride_x, int stride_y, Tile* dev_tiles, int* dev_tileTriCount)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index < numTiles)
	{
		for (int i = 0; i < dev_tileTriCount[index]; i++)
		{
			dev_tiles[index].triIndices[i] = 0;
		}
		dev_tileTriCount[index] = 0;
	}
}

void rasterizeTileBased(int w, int h, int numTriangles, Tile* dev_tiles, Primitive* dev_primitives,
						Fragment* dev_fragments, int* dev_depthBuffer, int* dev_mutex)
{
	int stride_x = glm::floor(w / numTilesX);
	int stride_y = glm::floor(h / numTilesY);
	int numTiles = (numTilesX+1)*(numTilesY+1);
	dim3 numThreadsPerBlock(128);

	//reset tile triangles
	dim3 blockCount1d_tiles(((numTiles - 1) / numThreadsPerBlock.x) + 1);
	resetTiles <<<blockCount1d_tiles, numThreadsPerBlock>>> (numTiles, stride_x, stride_y, 
															 dev_tiles, dev_tileTriCount);

	//preprocess step looping over all triangles to bin them into buckets corresponding to the tiles
	dim3 blockCount1d_triangles(((numTriangles - 1) / numThreadsPerBlock.x) + 1);
	bucketPrims_TileMutex <<<blockCount1d_triangles, numThreadsPerBlock >>> (w, stride_x, stride_y,
																			 numTriangles, dev_primitives,
																			 dev_tiles, dev_tileTriCount, 
																			 dev_tilemutex);

	int sideLength2d = 8;
	dim3 blockSize2d(sideLength2d, sideLength2d);

	int tilesX = glm::ceil(w / stride_x);
	int tileXcount = 0;
	int tileYcount = 0;
	for (int i = 0; i < w; i+=stride_y)
	{
		for (int j = 0; j < h; j+=stride_x)
		{		
			//Launch as many kernels as their are tiles			
			int tileID = tileXcount + tileYcount*(tilesX);
			//Each kernel is launched for the pixels contatined within it
			glm::ivec2 pixelMin = glm::ivec2(tileXcount*stride_x, tileYcount*stride_y);
			glm::ivec2 pixelMax = glm::ivec2(glm::min((tileXcount+1)*stride_x, w-1), 
											 glm::min((tileYcount+1)*stride_y, h-1));

			int numpixelsX = pixelMax.x - pixelMin.x;
			int numpixelsY = pixelMax.y - pixelMin.y;

			int pixelXoffset = tileXcount*stride_x;
			int pixelYoffset = tileYcount*stride_y;

			dim3 blockCount2d_tilePixels((numpixelsX - 1) / blockSize2d.x + 1,
										 (numpixelsY - 1) / blockSize2d.y + 1);
			RasterizePixels <<<blockCount2d_tilePixels, blockSize2d >>> (pixelXoffset, pixelYoffset,
																		 numpixelsX, numpixelsY, w,
																		 tileID, dev_tiles, 
																		 dev_primitives,
																		 dev_fragments,
																		 dev_tileTriCount,
																		 dev_depthBuffer, dev_mutex);
			checkCUDAError("tile rasterization failed");

			tileXcount++;
		}
		tileXcount = 0;
		tileYcount++;		
	}
}

//Perform rasterization.
void rasterize(uchar4 *pbo, const glm::mat4 & MVP, const glm::mat4 & MV, const glm::mat3 MV_normal) 
{
    int sideLength2d = 8;
    dim3 blockSize2d(sideLength2d, sideLength2d);
    dim3 blockCount2d((width  - 1) / blockSize2d.x + 1,
					  (height - 1) / blockSize2d.y + 1);
	
	//------------------------------------------------
	//Timer Start
	//timeStartCpu = std::chrono::high_resolution_clock::now();
	//------------------------------------------------

	//----------------------------------------------------------
	//----------------- Rasterization pipeline------------------
	//----------------------------------------------------------
	// Vertex Process & primitive assembly

	{
		curPrimitiveBeginId = 0;
		dim3 numThreadsPerBlock(128);

		auto it = mesh2PrimitivesMap.begin();
		auto itEnd = mesh2PrimitivesMap.end();

		for (; it != itEnd; ++it) 
		{
			auto p = (it->second).begin();	// each primitive
			auto pEnd = (it->second).end();
			for (; p != pEnd; ++p) 
			{
				dim3 numBlocksForVertices((p->numVertices + numThreadsPerBlock.x - 1) / numThreadsPerBlock.x);
				dim3 numBlocksForIndices((p->numIndices + numThreadsPerBlock.x - 1) / numThreadsPerBlock.x);

				_vertexTransformAndAssembly <<<numBlocksForVertices, numThreadsPerBlock>>>(p->numVertices, *p, MVP, MV,
																							MV_normal, width, height);
				checkCUDAError("Vertex Processing");
				cudaDeviceSynchronize();
				_primitiveAssembly <<<numBlocksForIndices, numThreadsPerBlock>>> (p->numIndices, curPrimitiveBeginId, 
																						dev_primitives, *p);
				checkCUDAError("Primitive Assembly");

				curPrimitiveBeginId += p->numPrimitives;
			}
		}

		checkCUDAError("Vertex Processing and Primitive Assembly");
	}
	
	numActivePrimitives = totalNumPrimitives;
#if BACKFACE_CULLING
	glm::vec3 camForward = glm::vec3(1.0f,1.0f,1.0f);
	BackFaceCulling(numActivePrimitives, dev_primitives, camForward);
#endif

	//reset fragment buffer
	cudaMemset(dev_fragmentBuffer, 0, width * height * sizeof(Fragment));
	//Reset Depth Buffer and mutex lock for depth buffer
	cudaMemset(dev_mutex, 0, width * height * sizeof(int)); //mutex for depth buffer
	initDepth <<<blockCount2d, blockSize2d >>>(width, height, dev_depth);
	
	dim3 numThreadsPerBlock(128);
#if SCANLINE
	// rasterize --> looping over all primitives(triangles)
	dim3 blockSize1d((numActivePrimitives - 1) / numThreadsPerBlock.x + 1);
	_rasterizeScanLine <<<blockSize1d, numThreadsPerBlock>>>(width, height, numActivePrimitives,
															 dev_primitives, dev_fragmentBuffer,
															 dev_depth, dev_mutex);
	checkCUDAError("scanline rendering failed");
#endif

#if TILEBASED
	cudaMemset(dev_tilemutex, 0, maxNumTiles * sizeof(int)); //mutex for tileTriCount buffer
	cudaMemset(dev_tileTriCount, 0, maxNumTiles * sizeof(int));

	rasterizeTileBased(width, height, numActivePrimitives, dev_tiles, dev_primitives, dev_fragmentBuffer,
					   dev_depth, dev_mutex);
	checkCUDAError("tile based rendering failed");
#endif

    // Copy depthbuffer colors into framebuffer
	render <<<blockCount2d, blockSize2d >>>(width, height, dev_fragmentBuffer, dev_framebuffer);
	checkCUDAError("fragment shader");

	//------------------------------------------------
	//Timer End
	//timeEndCpu = std::chrono::high_resolution_clock::now();
	//std::chrono::duration<double, std::milli> duration = timeEndCpu - timeStartCpu;
	//prevElapsedTime = static_cast<decltype(prevElapsedTime)>(duration.count());
	//printf("%f\n", prevElapsedTime);
	//------------------------------------------------

    // Copy framebuffer into OpenGL buffer for OpenGL previewing
    sendImageToPBO<<<blockCount2d, blockSize2d>>>(pbo, width, height, dev_framebuffer);
    checkCUDAError("copy render result to pbo");
}

//Called once at the end of the program to free CUDA memory.
void rasterizeFree() 
{
    // deconstruct primitives attribute/indices device buffer

	auto it(mesh2PrimitivesMap.begin());
	auto itEnd(mesh2PrimitivesMap.end());
	for (; it != itEnd; ++it) {
		for (auto p = it->second.begin(); p != it->second.end(); ++p) {
			cudaFree(p->dev_indices);
			cudaFree(p->dev_position);
			cudaFree(p->dev_normal);
			cudaFree(p->dev_texcoord0);
			cudaFree(p->dev_diffuseTex);

			cudaFree(p->dev_verticesOut);
		}
	}

	////////////
	cudaFree(dev_tiles);
	dev_tiles = NULL;

	cudaFree(dev_tileTriCount);
	dev_tileTriCount = NULL;

	cudaFree(dev_tilemutex);
	dev_tilemutex = NULL;

    cudaFree(dev_primitives);
    dev_primitives = NULL;

	cudaFree(dev_fragmentBuffer);
	dev_fragmentBuffer = NULL;

    cudaFree(dev_framebuffer);
    dev_framebuffer = NULL;

	cudaFree(dev_depth);
	dev_depth = NULL;

	cudaFree(dev_mutex);
	dev_depth = NULL;

    checkCUDAError("rasterize Free");
}
