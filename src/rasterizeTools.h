/**
 * @file      rasterizeTools.h
 * @brief     Tools/utility functions for rasterization.
 * @authors   Yining Karl Li
 * @date      2012-2015
 * @copyright University of Pennsylvania
 */

#pragma once

#include <cmath>
#include <glm/glm.hpp>
#include <util/utilityCore.hpp>

struct AABB {
    glm::vec3 min;
    glm::vec3 max;
};

/**
 * Multiplies a glm::mat4 matrix and a vec4.
 */
__host__ __device__ static
glm::vec3 multiplyMV(glm::mat4 m, glm::vec4 v) {
    return glm::vec3(m * v);
}

// CHECKITOUT
/**
 * Finds the axis aligned bounding box for a given triangle.
 */
__host__ __device__ static
AABB getAABBForTriangle(const glm::vec3 tri[3]) 
{
    AABB aabb;
    aabb.min = glm::vec3(
            min(min(tri[0].x, tri[1].x), tri[2].x),
            min(min(tri[0].y, tri[1].y), tri[2].y),
            min(min(tri[0].z, tri[1].z), tri[2].z));
    aabb.max = glm::vec3(
            max(max(tri[0].x, tri[1].x), tri[2].x),
            max(max(tri[0].y, tri[1].y), tri[2].y),
            max(max(tri[0].z, tri[1].z), tri[2].z));
    return aabb;
}

// CHECKITOUT
/**
 * Calculate the signed area of a given triangle.
 */
__host__ __device__ static
float calculateSignedArea(const glm::vec3 tri[3]) 
{
    return 0.5 * ((tri[2].x - tri[0].x) * (tri[1].y - tri[0].y) - (tri[1].x - tri[0].x) * (tri[2].y - tri[0].y));
}

// CHECKITOUT
/**
 * Helper function for calculating barycentric coordinates.
 */
__host__ __device__ static
float calculateBarycentricCoordinateValue(glm::vec2 a, glm::vec2 b, glm::vec2 c, const glm::vec3 tri[3]) {
    glm::vec3 baryTri[3];
    baryTri[0] = glm::vec3(a, 0);
    baryTri[1] = glm::vec3(b, 0);
    baryTri[2] = glm::vec3(c, 0);
    return calculateSignedArea(baryTri) / calculateSignedArea(tri);
}

// CHECKITOUT
/**
 * Calculate barycentric coordinates.
 */
__host__ __device__ static
glm::vec3 calculateBarycentricCoordinate(const glm::vec3 tri[3], glm::vec2 point) {
    float beta  = calculateBarycentricCoordinateValue(glm::vec2(tri[0].x, tri[0].y), point, glm::vec2(tri[2].x, tri[2].y), tri);
    float gamma = calculateBarycentricCoordinateValue(glm::vec2(tri[0].x, tri[0].y), glm::vec2(tri[1].x, tri[1].y), point, tri);
    float alpha = 1.0 - beta - gamma;
    return glm::vec3(alpha, beta, gamma);
}

// CHECKITOUT
/**
 * Check if a barycentric coordinate is within the boundaries of a triangle.
 */
__host__ __device__ static
bool isBarycentricCoordInBounds(const glm::vec3 barycentricCoord) {
    return barycentricCoord.x >= 0.0 && barycentricCoord.x <= 1.0 &&
           barycentricCoord.y >= 0.0 && barycentricCoord.y <= 1.0 &&
           barycentricCoord.z >= 0.0 && barycentricCoord.z <= 1.0;
}

// CHECKITOUT
/**
 * For a given barycentric coordinate, compute the corresponding z position
 * (i.e. depth) on the triangle.
 */
__host__ __device__ static
float getZAtCoordinate(const glm::vec3 barycentricCoord, const glm::vec3 tri[3]) {
    return -(barycentricCoord.x * tri[0].z
           + barycentricCoord.y * tri[1].z
           + barycentricCoord.z * tri[2].z);
}

//For edge function rasterization --> the actual technique used in mordern GPUs thats much faster than scanline
__host__ __device__ static
bool edgeFunction(const glm::vec3 &a, const glm::vec3 &b, const glm::vec2 &p)
{
	//the dot product thing commented below is essentially what the function 
	//is doing but the dot product is something i can easily understand at a glance
	//glm::vec3 v1 = b-a;
	//glm::vec3 v2 = p-a;	
	//return (glm::dot(v1, v2)>=0);

	return ((p.x - a.x) * (b.y - a.y) - (p.y - a.y) * (b.x - a.x) >= 0);
}

__host__ __device__ static
bool IsPointInsideTriangle(const glm::vec3 &V0, const glm::vec3 &V1, const glm::vec3 &V2, const glm::vec2 &p)
{
	bool inside = true;
	inside &= edgeFunction(V0, V1, p);
	inside &= edgeFunction(V1, V2, p);
	inside &= edgeFunction(V2, V0, p);

	return inside;
}
