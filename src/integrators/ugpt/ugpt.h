/*
    This file is part of Mitsuba, a physically based rendering system.

    Copyright (c) 2007-2014 by Wenzel Jakob and others.

    Mitsuba is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License Version 3
    as published by the Free Software Foundation.

    Mitsuba is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program. If not, see <http://www.gnu.org/licenses/>.
 */

#if !defined(__UGPT_H)
#define __UGPT_H

#include <mitsuba/mitsuba.h>

MTS_NAMESPACE_BEGIN


/* ==================================================================== */
/*                         Configuration storage                        */
/* ==================================================================== */


/// Configuration for the gradient path tracer.
struct UnstructuredGradientPathTracerConfig {
    int m_maxDepth;
    int m_minDepth;
    int m_rrDepth;
    bool m_strictNormals;
    Float m_shiftThreshold;
    bool m_reconstructL1;
    bool m_reconstructL2;
    Float m_reconstructAlpha;
    int m_nJacobiIters;
    int m_minMergeDepth;
    int m_maxMergeDepth;
};



/* ==================================================================== */
/*                         Integrator                         */

/* ==================================================================== */

struct PathNode
{
    struct Neighbor
    {
        PathNode* node; // We can use pointers here because PathNode structure is fixed when we set neighbors.
        Spectrum grad; // Gradient to that neighbor.
        Neighbor(PathNode* node) : node(node), grad(Spectrum(Float(0))) {}
    };
    RayDifferential lastRay; // ray before intersection
    Intersection its; // intersection info
    Point2 bsdfSample; // bsdf sample for next ray
    Spectrum weight; // accumulated weight before intersection
    Spectrum accumRad; // accumulated radiance from the camera ray before intersection
    Spectrum estRad[2]; // estimated radiance from the ray before intersection
    // We need 2 spectrum vectors to perform Jacobi iterations.
    std::vector<Neighbor> neighbors; 
    
    PathNode()
    {
        neighbors.reserve(5);
        accumRad = estRad[0] = estRad[1] = Spectrum(Float(0));
        weight = Spectrum(Float(1));
    }
};

struct PrecursorCacheInfo
{
    Point2 samplePos;
    Point2 apertureSample;
    Float timeSample;
    std::vector<PathNode> nodes;
    PrecursorCacheInfo()
    {
        nodes.reserve(5);
    }
    void clear()
    {
        nodes.clear();
    }
};

struct GradientMeshNode
{
    std::vector<int> neighbors;
    
};

class UnstructuredGradientPathIntegrator : public MonteCarloIntegrator {
public:
    UnstructuredGradientPathIntegrator(const Properties &props);

    /// Unserialize from a binary data stream
    UnstructuredGradientPathIntegrator(Stream *stream, InstanceManager *manager);


    /// Starts the rendering process.
    bool render(Scene *scene,
            RenderQueue *queue, const RenderJob *job,
            int sceneResID, int sensorResID, int samplerResID);


    /// Renders a block in the image.

    void serialize(Stream *stream, InstanceManager *manager) const;
    std::string toString() const;


    /// Used by Mitsuba for initializing sub-surface scattering.
    Spectrum Li(const RayDifferential &ray, RadianceQueryRecord &rRec) const;

    MTS_DECLARE_CLASS()

protected:
    
    std::vector<PrecursorCacheInfo> m_preCacheInfoList;
    
    void tracePrecursor(const Scene *scene, const Sensor *sensor, Sampler *sampler);
    
    void decideNeighbors(const Scene *scene, const Sensor *sensor);
    
    void traceDiff(const Scene *scene, const Sensor *sensor, Sampler *sampler);
    
    void communicateBidirectionalDiff(const Scene *scene);
    
    void iterateJacobi(const Scene *scene);
    
    void setOutputBuffer(const Scene *scene, Sensor *sensor);
private:
    UnstructuredGradientPathTracerConfig m_config;
    
    struct PointCloud {
        std::vector<PathNode*> nodes;
        inline size_t kdtree_get_point_count() const {
            return nodes.size();
        }

        inline float kdtree_distance(const float* p1, const size_t idx_p2, size_t) const {
            const float d0 = p1[0] - nodes[idx_p2]->its.p.x;
            const float d1 = p1[1] - nodes[idx_p2]->its.p.y;
            const float d2 = p1[2] - nodes[idx_p2]->its.p.z;
            return d0 * d0 + d1 * d1 + d2 * d2;
        }

        inline float kdtree_get_pt(const size_t idx, int dim) const {
            return nodes[idx]->its.p[dim];
        }

        template <class BBOX>
        bool kdtree_get_bbox(BBOX& /* bb */) const {
            return false;
        }
    };
    
    PointCloud m_pc;

};


MTS_NAMESPACE_END

#endif /* __GBDPT_H */
