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
#if defined(MTS_OPENMP)
#include <omp.h>
#endif

#if defined(MTS_OPENMP)
#define NANOFLANN_USE_OMP
#endif
#include "nanoflann.hpp"


//#define DUMP_GRAPH // dump graph structure as graph.txt if defined
#define PRINT_TIMING // print out timing info if defined
#define CACHE_FRIENDLY_ITERATOR // use cache friendly iterator
#define GDPT_STYLE_1ST_BOUNCE // use accumulated GDPT gradient
#define USE_FILTERS // use filters for neighbor connection
#define ADAPTIVE_DIFF_SAMPLING // use branching for diff samples
#define BACK_PROP_GRAD // back propagate second bounce gradient to first bounce.
#define SKIP_OVER_SPECULAR_NODES // skip over specular nodes
#define RECORD_VARIANCE

//#define USE_ADAPTIVE_WEIGHT // adaptive weights for neighbors based on feature similarity
//#define USE_RECON_RAYS // use lazy update for indirect light path radiance cache
//#define FACTOR_MATERIAL // use material factorization
//#define GRAD_IMPORTANCE_SAMPLING
//#define ADAPTIVE_GRAPH_SAMPLING // allocate samples based on graph connectivity

#define N_NEIGHBORS_TO_LOOKUP 4
#define N_MAX_NEIGHBORS 8

MTS_NAMESPACE_BEGIN

const Float D_EPSILON = std::numeric_limits<Float>::min();

/* ==================================================================== */
/*                         Configuration storage                        */
/* ==================================================================== */


/// Configuration for the gradient path tracer.
struct AdaptiveGradientPathTracerConfig {
    int m_maxDepth;
    int m_minDepth;
    int m_rrDepth;
    bool m_strictNormals;
    Float m_shiftThreshold;
    Float m_reconstructAlpha;
    int m_nJacobiIters;
    int m_minMergeDepth;
    int m_maxMergeDepth;
    int m_batchSize;
    bool m_usePixelNeighbors;
};

/* ==================================================================== */
/*                         Integrator                         */

/* ==================================================================== */

class AdaptiveGradientPathIntegrator : public MonteCarloIntegrator {
public:
    AdaptiveGradientPathIntegrator(const Properties &props);

    /// Unserialize from a binary data stream
    AdaptiveGradientPathIntegrator(Stream *stream, InstanceManager *manager);


    /// Starts the rendering process.
    bool render(Scene *scene,
            RenderQueue *queue, const RenderJob *job,
            int sceneResID, int sensorResID, int samplerResID);


    /// Renders a block in the image.

    void serialize(Stream *stream, InstanceManager *manager) const;
    std::string toString() const;


    /// Used by Mitsuba for initializing sub-surface scattering.
    Spectrum Li(const RayDifferential &ray, RadianceQueryRecord &rRec) const;

    virtual void cancel() {
        m_cancelled = true;
    }

    MTS_DECLARE_CLASS()

protected:
    bool m_cancelled;

    void tracePrecursor(const Scene *scene, const Sensor *sensor, Sampler *sampler);

    void decideNeighbors(const Scene *scene, const Sensor *sensor);

    void traceDiff(const Scene *scene, const Sensor *sensor, Sampler *sampler);
    
#if defined(ADAPTIVE_GRAPH_SAMPLING)
    void getMaxBlendingNum(const Scene *scene);
#endif
    
    void communicateBidirectionalDiff(const Scene *scene) 
    {
        for(int i=m_config.m_minMergeDepth; i<=m_config.m_maxMergeDepth; i++)
            communicateBidirectionalDiff(scene, i);
    }
    
    void communicateBidirectionalDiff(const Scene *scene, int depth);

    void iterateJacobi(const Scene *scene, const Sensor *sensor);

    void setOutputBuffer(const Scene *scene, Sensor *sensor, int batchSize, int iter);

    enum NeighborMethod {
        NEIGHBOR_RADIUS, NEIGHBOR_KNN
    } neighborMethod;

    /// Classification of vertices into diffuse and glossy.

    enum VertexType {
        VERTEX_TYPE_GLOSSY, ///< "Specular" vertex that requires the half-vector duplication shift.
        VERTEX_TYPE_DIFFUSE ///< "Non-specular" vertex that is rough enough for the reconnection shift.
    };
    /// Returns the vertex type of a vertex by its roughness value.

    struct BSDFSampleResult {
        BSDFSamplingRecord bRec; ///< The corresponding BSDF sampling record.
        Spectrum weight; ///< BSDF weight of the sampled direction.
        Float pdf; ///< PDF of the BSDF sample.
    };

    struct PathNode {
        RayDifferential lastRay; // ray before intersection
        Intersection its; // intersection info
        VertexType vertexType; // bsdf type of the vertex
        Point3 bsdfSample; // 3d full bsdf sample for generating the next ray
        Float rrSample; // Rassian roulette sample
        Spectrum weight_multiplier; // weight multiplier that happened in between current node and previous node
        Spectrum current_weight; // multiplied weight before arriving at this node
        Spectrum direct_lighting; // estimated direct lighting for the node

        Spectrum estRadBuffer[2]; // estimated radiance buffer for Jacobi iterations
        Spectrum estRad; // estimated radiance for the ray before intersection
        Vector3f color;
        // We need 2 spectrum vectors to perform Jacobi iterations.
        int sampleCount; // number of samples for branching
        int graph_index; // index of the corresponding vertex in boost graph
        int maxBlendingNum; // maximum number of nodes that can be used for blending
        
        int getSamplingRate() const {
#if defined(ADAPTIVE_GRAPH_SAMPLING)
            const int filter_size = 25;
            int sr = sqrt(maxBlendingNum) >= filter_size ? 1 : filter_size / sqrt(maxBlendingNum);
            return sr;
#else
            return 1;
#endif
        }

        struct Neighbor {
            PathNode* node; // We can use pointers here because PathNode structure is fixed when we set neighbors.
            Spectrum grad; // Gradient to the neighbor.
            Spectrum weight; // Weight for this connection. We set this to 1 for now.
            int sampleCount; // number of samples for gradient
            int index; // index indicating neighbor's direction for 1st bounce
#if defined(BACK_PROP_GRAD)
            Spectrum grad_before_conn; // gradient accumulated before connection happens
            bool merged; // indicates whether this neighbor gets merged successfully
            
            Neighbor(PathNode* node) : node(node), grad(Spectrum(Float(0))), 
            weight(Spectrum(Float(0))), sampleCount(1), merged(true), grad_before_conn(Spectrum(Float(0))) {
            }
#else
            Neighbor(PathNode* node) : node(node), grad(Spectrum(Float(0))), 
            weight(Spectrum(Float(0))), sampleCount(1) {
            }
#endif
            

            Neighbor() {
            }

            bool operator<(const Neighbor& n) const {
                return node < n.node;
            }

            bool operator==(const Neighbor& n) const {
                return node == n.node;
            }
        };
        std::vector<Neighbor> neighbors; // neighbors of this node
        bool addNeighbor(PathNode* neighbor, int neighbor_index = 0, bool filter = true);
        BSDFSampleResult getBSDFSampleResult() const;

#if defined(MTS_OPENMP)
        omp_lock_t writelock;
#endif

        PathNode() {
            neighbors.reserve(5);
            estRadBuffer[0] = estRadBuffer[1] = direct_lighting = Spectrum(Float(0));
            estRad = Spectrum(Float(0));
            weight_multiplier = current_weight = Spectrum(Float(1));
            sampleCount = 1;
            bsdfSample = Point3(-1.f, -1.f, -1.f);
            maxBlendingNum = 1;
#if defined(MTS_OPENMP)
            omp_init_lock(&writelock);
#endif
        }
#if defined(MTS_OPENMP)

        ~PathNode() {
            omp_destroy_lock(&writelock);
        }
#endif
    };

    struct PrecursorCacheInfo {
        int index;
        Point2 samplePos;
        Point2 apertureSample;
        Float timeSample;
        Spectrum very_direct_lighting;
        Spectrum factor;
        int maxBlendingNum;
        std::vector<PathNode> nodes;

        PrecursorCacheInfo() {
            nodes.reserve(5);
            factor = Spectrum(Float(1));
        }

        void clear() {
            nodes.clear();
        }
    };

    AdaptiveGradientPathTracerConfig m_config;

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

        inline void addGradient(const Spectrum& mainContrib, const Spectrum& shiftContrib, Float weight, bool hitEmitter = false) {
            Spectrum color = (mainContrib - shiftContrib) * weight;
            neighbor->grad += color / Spectrum(Float(neighbor->node->getSamplingRate())) / Float(neighbor->sampleCount);
#if defined(BACK_PROP_GRAD)
            if(hitEmitter) return;
            if(connection_status == RAY_NOT_CONNECTED && alive)
                neighbor->grad_before_conn += color / Spectrum(Float(neighbor->node->getSamplingRate())) / Float(neighbor->sampleCount);
#endif
        }

        inline Spectrum getCurrentWeight() const {
            return neighbor->node->current_weight;
        }
        
        inline Float getImportance() const {
            Float numerator = (main_throughput - throughput).abs().max();
            Float m = main_throughput.max();
            Float s = throughput.max();
            Float denominator = std::max(m, s);
            return numerator / (denominator+D_EPSILON);
        }

        RayDifferential ray; ///< Current ray.
        Spectrum throughput; ///< Current throughput of the path.
        Float pdf; ///< Current PDF of the main path as if the shifted path was actively sampled.
        // Note: Instead of storing throughput and pdf, it is possible to store Veach-style weight (throughput divided by pdf), if relative PDF (offset_pdf divided by base_pdf) is also stored. This might be more stable numerically.
        RadianceQueryRecord rRec; ///< The radiance query record for this ray.
        Intersection its;
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
        eta(1.0f),
        pdf(1.0f),
        throughput(Spectrum(1.0f)),
        lastNode_throughput(Spectrum(1.0f)),
        lastNode_pdf(1.0f),
        accumulateRadiance(true) {
        }

        /// Adds radiance to the ray.

        inline void multiply(const BSDFSampleResult& mainBsdfResult) {
            Spectrum throughput = mainBsdfResult.weight * mainBsdfResult.pdf;
            this->throughput *= throughput;
            this->pdf *= mainBsdfResult.pdf;

            eta *= mainBsdfResult.bRec.eta;
            if (rRec.depth >= pci->nodes.size()) {
                lastNode_throughput *= throughput;
                lastNode_pdf *= mainBsdfResult.pdf;
            }
        }

        inline void multiplyPDF(const Float& pdf) {
            this->pdf *= pdf;
            if (rRec.depth - 1 >= pci->nodes.size()) {
                lastNode_pdf *= pdf;
            }
        }

        inline void addRadiance(const Spectrum& estimated_radiance) {
            if(!accumulateRadiance) return;
            if (rRec.depth < pci->nodes.size()) {
                pci->nodes[rRec.depth - 1].direct_lighting += estimated_radiance / Float(pci->nodes.front().getSamplingRate());
            } else {
                pci->nodes.back().estRad += lastNode_throughput * estimated_radiance / lastNode_pdf / Float(pci->nodes.front().getSamplingRate());
                pci->nodes.back().estRadBuffer[0] = pci->nodes.back().estRad; // set initial values
            }
        }

        bool accumulateRadiance;
        RayDifferential ray; ///< Current ray.

        Spectrum throughput; ///< Current throughput of the path.
        Float pdf; ///< Current PDF of the path.

        // Note: Instead of storing throughput and pdf, it is possible to store Veach-style weight (throughput divided by pdf), if relative PDF (offset_pdf divided by base_pdf) is also stored. This might be more stable numerically.
        Spectrum lastNode_throughput;
        Float lastNode_pdf;

        RadianceQueryRecord rRec; ///< The radiance query record for this ray.
        Float eta; ///< Current refractive index of the ray.
        PrecursorCacheInfo* pci; // Cached information from precursor
        void spawnShiftedRay(std::vector<ShiftedRayState>& shiftedRays); // spawns shifted rays for current depth

    };


    struct BranchArguments
    {
        std::vector<ShiftedRayState> shiftedRays;
    };
    void evaluateDiff(MainRayState& main, BranchArguments* branchArguments = NULL);
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

        if (!(result.pdf <= (Float) 0 || fabs(result.bRec.wo.length() - 1.0) < 0.01)) {
            printf("%f %f\n", sample.x, sample.y);
            printf("%f\n", result.bRec.wo.length());
            result.pdf = 0;
            return result;
        }

        SAssert(result.pdf <= (Float) 0 || fabs(result.bRec.wo.length() - 1.0) < 0.01);
        return result;
    }
    
    inline BSDFSampleResult sampleBSDF(MainRayState& rayState, const Point3& sample) {
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

        if (!(result.pdf <= (Float) 0 || fabs(result.bRec.wo.length() - 1.0) < 0.01)) {
            printf("%f %f\n", sample.x, sample.y);
            printf("%f\n", result.bRec.wo.length());
            result.pdf = 0;
            return result;
        }

        SAssert(result.pdf <= (Float) 0 || fabs(result.bRec.wo.length() - 1.0) < 0.01);
        return result;
    }

    VertexType getVertexTypeByRoughness(Float roughness, const AdaptiveGradientPathTracerConfig& config) {
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

    VertexType getVertexType(const BSDF* bsdf, Intersection& its, const AdaptiveGradientPathTracerConfig& config, unsigned int bsdfType);

    VertexType getVertexType(MainRayState& ray, const AdaptiveGradientPathTracerConfig& config, unsigned int bsdfType);

    VertexType getVertexType(ShiftedRayState& ray, const AdaptiveGradientPathTracerConfig& config, unsigned int bsdfType);

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
            Float sqr_dist = d0 * d0 + d1 * d1 + d2 * d2;
#if defined(USE_NORMAL_NN)
            const Float n0 = p1[3] - nodes[idx_p2]->its.geoFrame.n.x;
            const Float n1 = p1[4] - nodes[idx_p2]->its.geoFrame.n.y;
            const Float n2 = p1[5] - nodes[idx_p2]->its.geoFrame.n.z;
            sqr_dist += (n0 * n0 + n1 * n1 + n2 * n2);
#endif
            return sqr_dist;
        }

        inline Float kdtree_get_pt(const size_t idx, int dim) const {
            if (dim < 3)
                return nodes[idx]->its.p[dim];
            return nodes[idx]->its.geoFrame.n[dim - 3];
        }

        template <class BBOX>
        bool kdtree_get_bbox(BBOX& /* bb */) const {
            return false;
        }
    };

    std::shared_ptr<PointCloud> m_pc;
    std::shared_ptr<std::vector<PrecursorCacheInfo> > m_preCacheInfoList;

    typedef nanoflann::KDTreeSingleIndexAdaptor<
    nanoflann::L2_Simple_Adaptor<Float, PointCloud>,
    PointCloud, 3 /* dim */ > kd_tree_t;
    enum PrecursorTask{PRECURSOR_GET_FACTOR, PRECURSOR_LOOP} m_precursorTask;
    
#if defined(RECORD_VARIANCE)
    std::vector<Spectrum> buffer_throughput;
    std::vector<Spectrum> buffer_dx;
    std::vector<Spectrum> buffer_dy;
    std::vector<Spectrum> buffer_var_throughput;
    std::vector<Spectrum> buffer_var_dx;
    std::vector<Spectrum> buffer_var_dy;
#endif

#if defined(USE_RECON_RAYS)

    enum CurrentMode {
        SAMPLE_MODE, RECON_MODE
    } m_currentMode;
    std::shared_ptr<std::vector<PrecursorCacheInfo> > m_preCacheInfoListBuffer;
    std::shared_ptr<PointCloud> m_pcBuffer;
    std::shared_ptr<kd_tree_t> m_treeBuffer;
#endif
};


MTS_NAMESPACE_END

#endif /* __GBDPT_H */
