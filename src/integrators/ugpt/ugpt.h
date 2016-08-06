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
    Float m_reconstructAlpha;
    int m_nJacobiIters;
    int m_minMergeDepth;
    int m_maxMergeDepth;
};



/* ==================================================================== */
/*                         Integrator                         */

/* ==================================================================== */

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
    
    void tracePrecursor(const Scene *scene, const Sensor *sensor, Sampler *sampler);

    void decideNeighbors(const Scene *scene, const Sensor *sensor);

    void traceDiff(const Scene *scene, const Sensor *sensor, Sampler *sampler);

    void communicateBidirectionalDiff(const Scene *scene);

    void iterateJacobi(const Scene *scene);

    void setOutputBuffer(const Scene *scene, Sensor *sensor);

    struct PathNode {

        struct Neighbor {
            PathNode* node; // We can use pointers here because PathNode structure is fixed when we set neighbors.
            Spectrum grad; // Gradient to that neighbor.

            Neighbor(PathNode* node) : node(node), grad(Spectrum(Float(0))) {
            }
        };
        RayDifferential lastRay; // ray before intersection
        Intersection its; // intersection info
        Point2 bsdfSample; // 2d bsdf sample for generating the next ray
        Spectrum weight; // accumulated weight before intersection
        Spectrum accumRad; // accumulated radiance for the camera ray before intersection
        Spectrum estRad[2]; // estimated radiance for the ray before intersection
        // We need 2 spectrum vectors to perform Jacobi iterations.
        std::vector<Neighbor> neighbors; // neighbors of this node

        PathNode() {
            neighbors.reserve(5);
            accumRad = estRad[0] = estRad[1] = Spectrum(Float(0));
            weight = Spectrum(Float(1));
        }
    };

    struct PrecursorCacheInfo {
        Point2 samplePos;
        Point2 apertureSample;
        Float timeSample;
        std::vector<PathNode> nodes;

        PrecursorCacheInfo() {
            nodes.reserve(5);
        }

        void clear() {
            nodes.clear();
        }
    };

    std::vector<PrecursorCacheInfo> m_preCacheInfoList;

    UnstructuredGradientPathTracerConfig m_config;

    enum RayConnection {
        RAY_NOT_CONNECTED, ///< Not yet connected - shifting in progress.
        RAY_RECENTLY_CONNECTED, ///< Connected, but different incoming direction so needs a BSDF evaluation.
        RAY_CONNECTED ///< Connected, allows using BSDF values from the base path.
    };

    struct ShiftedRayState {

        ShiftedRayState()
        :
        eta(1.0f),
        pdf(1.0f),
        throughput(Spectrum(1.0f)),
        main_pdf(1.0f),
        main_throughput(Spectrum(1.0f)),
        alive(true),
        activeDepth(0),
        connection_status(RAY_NOT_CONNECTED) {
        }
        /// Adds gradient to the ray.

        inline void addGradient(const Spectrum& mainContrib, const Spectrum& shiftContrib, Float weight) {
            Spectrum color = (mainContrib - shiftContrib) * weight;
            neighbor->grad += color;
        }

        RayDifferential ray; ///< Current ray.
        Spectrum throughput; ///< Current throughput of the path.
        Float pdf; ///< Current PDF of the path.
        // Note: Instead of storing throughput and pdf, it is possible to store Veach-style weight (throughput divided by pdf), if relative PDF (offset_pdf divided by base_pdf) is also stored. This might be more stable numerically.
        RadianceQueryRecord rRec; ///< The radiance query record for this ray.
        Float eta; ///< Current refractive index of the ray.
        bool alive; ///< Whether the path matching to the ray is still good. Otherwise it's an invalid offset path with zero PDF and throughput.

        RayConnection connection_status; ///< Whether the ray has been connected to the base path, or is in progress.
        PathNode::Neighbor* neighbor; // neighbor connection related to a shifted ray
        int activeDepth; // active depth of a shifted ray
        Spectrum main_throughput; // record main throughput starting from activeDepth for shifted samples
        Float main_pdf; // record main pdf starting from activeDepth for shifted samples
    };

    struct MainRayState {

        MainRayState()
        :
        radiance(0.0f),
        eta(1.0f),
        pdf(1.0f),
        throughput(Spectrum(1.0f)) {
        }

        /// Adds radiance to the ray.

        inline void addRadiance(const Spectrum& contribution, Float weight) {
            Spectrum color = contribution * weight;
            radiance += color;
        }

        RayDifferential ray; ///< Current ray.

        Spectrum throughput; ///< Current throughput of the path.
        Float pdf; ///< Current PDF of the path.

        // Note: Instead of storing throughput and pdf, it is possible to store Veach-style weight (throughput divided by pdf), if relative PDF (offset_pdf divided by base_pdf) is also stored. This might be more stable numerically.

        Spectrum radiance; ///< Radiance accumulated so far.

        RadianceQueryRecord rRec; ///< The radiance query record for this ray.
        Float eta; ///< Current refractive index of the ray.
        PrecursorCacheInfo* pci; // Cached information from precursor

        void spawnShiftedRay(std::vector<ShiftedRayState>& shiftedRays); // spawns shifted rays for current depth
    };



    void evaluateDiff(MainRayState& main, Spectrum& out_veryDirect);

    void evaluatePrecursor(MainRayState& main);








    /// Result of a reconnection shift.

    struct ReconnectionShiftResult {
        bool success; ///< Whether the shift succeeded.
        Float jacobian; ///< Local Jacobian determinant of the shift.
        Vector3 wo; ///< World space outgoing vector for the shift.
    };
    ReconnectionShiftResult environmentShift(const Scene* scene, const Ray& mainRay, Point3 shiftSourceVertex);

    /// Tries to connect the offset path to a specific vertex of the main path.

    ReconnectionShiftResult reconnectShift(const Scene* scene, Point3 mainSourceVertex, Point3 targetVertex, Point3 shiftSourceVertex, Vector3 targetNormal, Float time);

    struct BSDFSampleResult {
        BSDFSamplingRecord bRec; ///< The corresponding BSDF sampling record.
        Spectrum weight; ///< BSDF weight of the sampled direction.
        Float pdf; ///< PDF of the BSDF sample.
    };

    inline BSDFSampleResult sampleBSDF(MainRayState& rayState, const Point2& sample) {
        Intersection& its = rayState.rRec.its;
        RadianceQueryRecord& rRec = rayState.rRec;
        RayDifferential& ray = rayState.ray;

        // Note: If the base path's BSDF evaluation uses random numbers, it would be beneficial to use the same random numbers for the offset path's BSDF.
        //       This is not done currently.

        const BSDF* bsdf = its.getBSDF(ray);

        // Sample BSDF * cos(theta).
        BSDFSampleResult result = {
            BSDFSamplingRecord(its, rRec.sampler, ERadiance),
            Spectrum(),
            (Float) 0
        };

        result.weight = bsdf->sample(result.bRec, result.pdf, sample);

        // Variable result.pdf will be 0 if the BSDF sampler failed to produce a valid direction.

        SAssert(result.pdf <= (Float) 0 || fabs(result.bRec.wo.length() - 1.0) < 0.00001);
        return result;
    }

    /// Classification of vertices into diffuse and glossy.

    enum VertexType {
        VERTEX_TYPE_GLOSSY, ///< "Specular" vertex that requires the half-vector duplication shift.
        VERTEX_TYPE_DIFFUSE ///< "Non-specular" vertex that is rough enough for the reconnection shift.
    };
    /// Returns the vertex type of a vertex by its roughness value.

    VertexType getVertexTypeByRoughness(Float roughness, const UnstructuredGradientPathTracerConfig& config) {
        if (roughness <= config.m_shiftThreshold) {
            return VERTEX_TYPE_GLOSSY;
        } else {
            return VERTEX_TYPE_DIFFUSE;
        }
    }

    /// Returns the vertex type (diffuse / glossy) of a vertex, for the purposes of determining
    /// the shifting strategy.
    ///
    /// A bare classification by roughness alone is not good for multi-component BSDFs since they
    /// may contain a diffuse component and a perfect specular component. If the base path
    /// is currently working with a sample from a BSDF's smooth component, we don't want to care
    /// about the specular component of the BSDF right now - we want to deal with the smooth component.
    ///
    /// For this reason, we vary the classification a little bit based on the situation.
    /// This is perfectly valid, and should be done.

    VertexType getVertexType(const BSDF* bsdf, Intersection& its, const UnstructuredGradientPathTracerConfig& config, unsigned int bsdfType);

    VertexType getVertexType(MainRayState& ray, const UnstructuredGradientPathTracerConfig& config, unsigned int bsdfType);

    VertexType getVertexType(ShiftedRayState& ray, const UnstructuredGradientPathTracerConfig& config, unsigned int bsdfType);

    /// Result of a half vector shift.

    struct HalfVectorShiftResult {
        bool success; ///< Whether the shift succeeded.
        Float jacobian; ///< Local Jacobian determinant of the shift.
        Vector3 wo; ///< Tangent space outgoing vector for the shift.
    };
    HalfVectorShiftResult halfVectorShift(Vector3 tangentSpaceMainWi, Vector3 tangentSpaceMainWo, Vector3 tangentSpaceShiftedWi, Float mainEta, Float shiftedEta);

    inline BSDFSampleResult sampleBSDF(MainRayState& rayState) {
        RadianceQueryRecord& rRec = rayState.rRec;
        Point2 sample = rRec.nextSample2D();
        return sampleBSDF(rayState, sample);
    }

    struct PointCloud {
        std::vector<PathNode*> nodes;

        inline size_t kdtree_get_point_count() const {
            return nodes.size();
        }

        inline Float kdtree_distance(const Float* p1, const size_t idx_p2, size_t) const {
            const Float d0 = p1[0] - nodes[idx_p2]->its.p.x;
            const Float d1 = p1[1] - nodes[idx_p2]->its.p.y;
            const Float d2 = p1[2] - nodes[idx_p2]->its.p.z;
            return d0 * d0 + d1 * d1 + d2 * d2;
        }

        inline Float kdtree_get_pt(const size_t idx, int dim) const {
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
