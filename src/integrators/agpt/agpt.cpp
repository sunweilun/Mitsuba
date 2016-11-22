/*
    This file is part of Mitsuba, a physically based rendering system.

    Copyright (c) 2007-2012 by Wenzel Jakob and others.

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

#include <mitsuba/bidir/util.h>
#include <mitsuba/render/scene.h>
#include <mitsuba/core/statistics.h>
#include <mitsuba/render/renderproc.h>
#include "mitsuba/core/plugin.h"
#include "agpt.h"

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/connected_components.hpp>

MTS_NAMESPACE_BEGIN

        /*!\plugin{ugpt}{Unstructured Gradient-domain path tracer}
         * \order{20}
         * \parameters{
         *	   \parameter{reconstructL1}{\Boolean}{If set, the rendering method reconstructs the final image using a reconstruction method 
         *           that efficiently kills many image artifacts. The reconstruction is slightly biased, but the bias will go away by increasing sample count. \default{\code{true}}
         *     }
         *	   \parameter{reconstructL2}{\Boolean}{If set, the rendering method reconstructs the final image using a reconstruction method that is unbiased, 
         *			but sometimes introduces severe dipole artifacts. \default{\code{false}}
         *     }
         *	   \parameter{shiftThreshold}{\Float}{Specifies the roughness threshold for classifying materials as 'diffuse', in contrast to 'specular', 
         *			for the purposes of constructing paths pairs for estimating pixel differences. This value should usually be somewhere between 0.0005 and 0.01. 
         *			If the result image has noise similar to standard path tracing, increasing or decreasing this value may sometimes help. This implementation assumes that this value is small.\default{\code{0.001}}
         *	   }
         *	   \parameter{reconstructAlpha}{\Float}{	
         *			Higher value makes the reconstruction trust the noisy color image more, giving less weight to the usually lower-noise gradients. 
         *			The optimal value tends to be around 0.2, but for scenes with much geometric detail at sub-pixel level a slightly higher value such as 0.3 or 0.4 may be tried.\default{\code{0.2}}
                   }
         * }
         *
         *
         * This plugin implements a gradient-domain path tracer (short: G-PT) as described in the paper "Gradient-Domain Path Tracing" by Kettunen et al. 
         * It samples difference images in addition to the standard color image, and reconstructs the final image based on these.
         * It supports classical materials like diffuse, specular and glossy materials, and area and point lights, depth-of-field, and low discrepancy samplers. 
         * There is also experimental support for sub-surface scattering and motion blur. Note that this is still an experimental implementation of Gradient-Domain Path Tracing 
         * that has not been tested with all of Mitsuba's features. Notably there is no support yet for any kind of participating media or directional lights. 
         * Environment maps are supported, though. Does not support the 'hide emitters' option even though it is displayed.

         *
         */

        // Output buffer names.
        static const size_t BUFFER_FINAL = 0; ///< Buffer index for the final image. Also used for preview.
static const size_t BUFFER_THROUGHPUT = 1; ///< Buffer index for the noisy color image.
static const size_t BUFFER_DX = 2; ///< Buffer index for the X gradients.
static const size_t BUFFER_DY = 3; ///< Buffer index for the Y gradients.
static const size_t BUFFER_VERY_DIRECT = 4; ///< Buffer index for very direct light.


#if defined(PRINT_TIMING)
#include <sys/time.h>

class MyTimer {
protected:
    timeval ts, te;
public:

    void tic() {
        gettimeofday(&ts, NULL);
    }

    double toc() {
        gettimeofday(&te, NULL);
        return double(te.tv_sec - ts.tv_sec) + double(te.tv_usec - ts.tv_usec)*1e-6;
    }
};
#endif

inline Float fabs(const Float& a) {
    return (a > 0) ? a : -a;
}

inline size_t ceil(size_t num, size_t denom) {
    return 1 + (num - 1) / denom;
}

void AdaptiveGradientPathIntegrator::tracePrecursor(const Scene *scene, const Sensor *sensor, Sampler *sampler) {
    bool needsApertureSample = sensor->needsApertureSample();
    bool needsTimeSample = sensor->needsTimeSample();
    // Get ready for sampling.

    const int& cx = sensor->getFilm()->getCropSize().x;
    const int& cy = sensor->getFilm()->getCropSize().y;
    const int& bSize = scene->getBlockSize();
    const int& bx = ceil(cx, bSize);
    const int& by = ceil(cy, bSize);

#if defined(MTS_OPENMP)
    ref<Scheduler> sched = Scheduler::getInstance();
    const int& nCores = sched->getCoreCount();
    ref_vector<Sampler> samplers(nCores);
    for (int i = 0; i < nCores; i++)
        samplers[i] = sampler->clone();

#if defined(GDPT_STYLE_1ST_BOUNCE)
    RadianceQueryRecord r(scene, samplers.front());
    Point2 perturb = r.nextSample2D();
#endif

#pragma omp parallel for schedule(dynamic)
#endif
    for (int blockIndex = 0; blockIndex < bx * by; blockIndex++) {
#if defined(MTS_OPENMP)
        Sampler* sampler = samplers[omp_get_thread_num()];
#endif
        Point2 apertureSample(0.5f);
        Float timeSample = 0.5f;
        Float diffScaleFactor = 1.0f / std::sqrt((Float) sampler->getSampleCount());
        RadianceQueryRecord rRec(scene, sampler);

        for (int pointIndex = 0; pointIndex < bSize * bSize; pointIndex++) {
            int x = (blockIndex % bx) * bSize + pointIndex % bSize;
            int y = (blockIndex / bx) * bSize + pointIndex / bSize;
            if (x >= cx || y >= cy) continue;

            PrecursorCacheInfo &pci = (*m_preCacheInfoList)[y * cx + x];
            pci.clear(); // start over
            Point2i offset(x, y);
            sampler->generate(offset);
            rRec.newQuery(RadianceQueryRecord::ESensorRay, sensor->getMedium());

            Point2 samplePos;
#if defined(GDPT_STYLE_1ST_BOUNCE)
            samplePos = Point2(offset) + Vector2(perturb);
#else
            samplePos = Point2(offset) + Vector2(rRec.nextSample2D());
#endif
            rRec.nextSample2D();

            if (needsApertureSample) {
                apertureSample = rRec.nextSample2D();
            }
            if (needsTimeSample) {
                timeSample = rRec.nextSample1D();
            }
            pci.index = y * cx + x;
            pci.samplePos = samplePos;
            pci.apertureSample = apertureSample;
            pci.timeSample = timeSample;
            MainRayState mainRay;
            mainRay.throughput = sensor->sampleRayDifferential(mainRay.ray,
                    pci.samplePos, pci.apertureSample, pci.timeSample);
            mainRay.ray.scaleDifferential(diffScaleFactor);
            mainRay.pci = &pci;
            mainRay.rRec = rRec;
            mainRay.rRec.its = rRec.its;
            evaluatePrecursor(mainRay);
            pci.very_direct_lighting = Spectrum(Float(0));
        }
    }
}

bool similar(const Spectrum& c1, const Spectrum& c2, Float thres) {
    Float r1, g1, b1;
    c1.toLinearRGB(r1, g1, b1);
    Float r2, g2, b2;
    c2.toLinearRGB(r2, g2, b2);
    if (r1 * thres < r2 || r2 * thres < r1) return false;
    if (g1 * thres < g2 || g2 * thres < g1) return false;
    if (b1 * thres < b2 || b2 * thres < b1) return false;
    return true;
}

bool AdaptiveGradientPathIntegrator::PathNode::addNeighbor(PathNode* neighbor, int neighbor_index, bool filter) {
    if (filter) {
        // filter normal
        if (dot(neighbor->its.geoFrame.n, its.geoFrame.n) < 0.5f) return false;
        TVector3<Float> diff_vec = neighbor->its.p - its.p;
        Float len = diff_vec.length();
        if (fabs(dot(neighbor->its.geoFrame.n, diff_vec)) > 0.3f * len) return false;
        if (fabs(dot(its.geoFrame.n, diff_vec)) > 0.3f * len) return false;

        // filter # of neighbors
        if (neighbors.size() >= 4) return false;

        // filter glossy vertices
        //if (neighbor->vertexType == VERTEX_TYPE_GLOSSY) return false;
        //if (vertexType == VERTEX_TYPE_GLOSSY) return false;

        const Float color_thres = 2;
        const Float weight_thres = 4;
        
        Spectrum diff = its.getBSDF()->getDiffuseReflectance(its);
        Spectrum spec = its.getBSDF()->getSpecularReflectance(its);

        Spectrum n_diff = neighbor->its.getBSDF()->getDiffuseReflectance(neighbor->its);
        Spectrum n_spec = neighbor->its.getBSDF()->getSpecularReflectance(neighbor->its);

        // color filter
        if (!similar(diff, n_diff, color_thres)) return false;
        if (!similar(spec, n_spec, color_thres)) return false;
        
        // weight filter
        if(!similar(current_weight, neighbor->current_weight, weight_thres)) return false; 
        // this filter must be applied to make sure unbiasedness because radiance values of nodes with 0 weight may bias the gradient due to Russian Roullete
    }

    Float dist = distance(neighbor->its.p, its.p);

    neighbors.push_back(neighbor);
    neighbors.back().weight = Spectrum(Float(1));
    neighbors.back().index = neighbor_index;
    return true;
}

void AdaptiveGradientPathIntegrator::decideNeighbors(const Scene *scene, const Sensor *sensor) {
    if (m_cancelled) return;
    if (m_config.m_nJacobiIters == 0) return;
    neighborMethod = NEIGHBOR_KNN;

    size_t chunk_size = scene->getBlockSize();
    chunk_size *= chunk_size;
    const int& cx = sensor->getFilm()->getCropSize().x;
    const int& cy = sensor->getFilm()->getCropSize().y;


    for (int mergeDepth = m_config.m_minMergeDepth; mergeDepth <= m_config.m_maxMergeDepth; mergeDepth++) {

        if (mergeDepth == 0 && m_config.m_usePixelNeighbors) {

#if defined(MTS_OPENMP)
#pragma omp parallel for schedule(dynamic, chunk_size)
#endif
            for (size_t i = 0; i < (*m_preCacheInfoList).size(); i++) {
                if (!(*m_preCacheInfoList)[i].nodes[0].its.isValid()) continue;
                int x = i % cx;
                int y = i / cx;
                Vector2i center(x, y);
                std::vector<Vector2i> neighbors(4);
                neighbors[0] = Vector2i(center + Vector2i(-1, 0));
                neighbors[1] = Vector2i(center + Vector2i(1, 0));
                neighbors[2] = Vector2i(center + Vector2i(0, -1));
                neighbors[3] = Vector2i(center + Vector2i(0, 1));
                for (int k = 0; k < neighbors.size(); k++) {
                    const Vector2i &n = neighbors[k];
                    if (n.x >= cx || n.y >= cy || n.x < 0 || n.y < 0)
                        continue;
                    int j = n.y * cx + n.x;
                    if (!(*m_preCacheInfoList)[j].nodes[0].its.isValid()) continue;
#if defined(GDPT_STYLE_1ST_BOUNCE)
                    (*m_preCacheInfoList)[i].nodes[0].addNeighbor(&(*m_preCacheInfoList)[j].nodes[0], k, false);
#else
                    (*m_preCacheInfoList)[i].nodes[0].addNeighbor(&(*m_preCacheInfoList)[j].nodes[0], k, true);
#endif
                }
            }
            continue;
        }



        // build connection using spatial neighbors
        {
            std::shared_ptr<kd_tree_t> index;


#if defined(USE_RECON_RAYS)
            if (m_currentMode == RECON_MODE)
                m_pc.reset(new PointCloud());
#endif            
            (*m_pc).nodes.clear();

            for (auto& pci : (*m_preCacheInfoList)) {
                if (mergeDepth >= pci.nodes.size() || !pci.nodes[mergeDepth].its.isValid())
                    continue;
                (*m_pc).nodes.push_back(&pci.nodes[mergeDepth]);
            }

#if defined(USE_RECON_RAYS)

            if (m_currentMode == SAMPLE_MODE) {
                // collect nodes for mergeDepth

                m_pcBuffer = m_pc;
                index.reset(new kd_tree_t(3, *m_pcBuffer, nanoflann::KDTreeSingleIndexAdaptorParams()));
                index->buildIndex();
                m_treeBuffer = index;
            } else {
                index = m_treeBuffer;
#if defined(MTS_OPENMP)
#pragma omp parallel for schedule(dynamic)
#endif
                for (size_t base = 0; base < (*m_pcBuffer).nodes.size(); base += chunk_size) {
                    for (size_t k = 0; k < chunk_size; k++) {
                        size_t i = base + k;
                        if (i >= (*m_pcBuffer).nodes.size()) continue;
                        (*m_pcBuffer).nodes[i]->neighbors.clear(); // clear neighbors because the values are already filtered so we do not need them for reconstruction.
                    }
                }
            }
#else
            // collect nodes for mergeDepth
            index.reset(new kd_tree_t(3, *m_pc, nanoflann::KDTreeSingleIndexAdaptorParams()));
            index->buildIndex();
            std::shared_ptr<PointCloud> m_pcBuffer = m_pc;
#endif
            switch (neighborMethod) {
                case NEIGHBOR_RADIUS:
                {
                    const Float radius = 0.05f * 0.05f;
#if defined(MTS_OPENMP)
#pragma omp parallel for schedule(dynamic)
#endif
                    for (size_t i = 0; i < (*m_pc).nodes.size(); i++) {
                        Float* query_pt = (Float*) & (*m_pc).nodes[i]->its.p;
                        std::vector<std::pair<size_t, Float> > indices_dists;
                        nanoflann::RadiusResultSet<Float> resultSet(radius, indices_dists);
                        index->findNeighbors(resultSet, query_pt, nanoflann::SearchParams());
                        for (auto& item : indices_dists) {
                            if (item.first == i) continue; // exclude self
                            (*m_pc).nodes[i]->addNeighbor((*m_pc).nodes[item.first]);
                        }
                    }
                    break;
                }
                case NEIGHBOR_KNN:
                {
                    int nn = 4; // number of neighbors
#if defined(USE_RECON_RAYS)
                    if (m_currentMode == RECON_MODE) nn = 1;
#endif
#if defined(MTS_OPENMP)
#pragma omp parallel for schedule(dynamic, chunk_size)
#endif
                    for (size_t i = 0; i < (*m_pc).nodes.size(); i++) {
                        Float* query_pt = (Float*) & (*m_pc).nodes[i]->its.p;
                        std::vector<size_t> indices(nn);
                        std::vector<Float> dist_sqr(nn);
                        nanoflann::KNNResultSet<Float> resultSet(nn);
                        resultSet.init(indices.data(), dist_sqr.data());
                        index->findNeighbors(resultSet, query_pt, nanoflann::SearchParams(10, 0));
                        for (auto& neighbor_idx : indices) {
#if defined(USE_RECON_RAYS)
                            if (neighbor_idx == i && m_currentMode == SAMPLE_MODE) continue;
#else
                            if (neighbor_idx == i) continue; // exclude self
#endif
#if defined(MTS_OPENMP)
                            omp_set_lock(&(*m_pc).nodes[i]->writelock);
#endif

#if defined(USE_RECON_RAYS)     
                            bool unfiltered = (*m_pc).nodes[i]->addNeighborWithFilter((*m_pcBuffer).nodes[neighbor_idx]);
#else
                            bool unfiltered = (*m_pc).nodes[i]->addNeighbor((*m_pc).nodes[neighbor_idx]);
#endif

#if defined(MTS_OPENMP)
                            omp_unset_lock(&(*m_pc).nodes[i]->writelock);
#endif                                
                            if (!unfiltered) continue;
#if defined(MTS_OPENMP)                           
                            omp_set_lock(&(*m_pcBuffer).nodes[neighbor_idx]->writelock);
#endif
                            // make sure the graph is bidirectional, this may produce duplicated neighbors.
#if defined(USE_RECON_RAYS)
                            (*m_pcBuffer).nodes[neighbor_idx]->neighbors.push_back((*m_pc).nodes[i]); // the neighbors are the old ones in the buffer
#else
                            (*m_pc).nodes[neighbor_idx]->neighbors.push_back((*m_pc).nodes[i]);
#endif

#if defined(MTS_OPENMP)
                            omp_unset_lock(&(*m_pcBuffer).nodes[neighbor_idx]->writelock);
#endif
                        }
                    }
                    // take out duplicated neighbors
#if defined(MTS_OPENMP)
#pragma omp parallel for schedule(dynamic, chunk_size)
#endif
                    for (size_t i = 0; i < (*m_pc).nodes.size(); i++) {
                        auto& neighbors = (*m_pc).nodes[i]->neighbors;
                        std::sort(neighbors.begin(), neighbors.end());
                        auto it = std::unique(neighbors.begin(), neighbors.end());
                        neighbors.resize(it - neighbors.begin());
                    }
                    break;
                }
            }
        }
    }

#if defined(DUMP_GRAPH)
    FILE* file = fopen("graph.txt", "w");
    for (auto& pci : *m_preCacheInfoList) {
        auto& node = pci.nodes.back();
        fprintf(file, "p %f %f %f\n", node.its.p.x, node.its.p.y, node.its.p.z);
        for (auto& neighbor : node.neighbors) {
            fprintf(file, "l %f %f %f ", neighbor.node->its.p.x, neighbor.node->its.p.y, neighbor.node->its.p.z);
            fprintf(file, "%f %f %f\n", node.its.p.x, node.its.p.y, node.its.p.z);
        }
    }
    fclose(file);
#endif

}

#if defined(ADAPTIVE_GRAPH_SAMPLING)

void AdaptiveGradientPathIntegrator::getMaxBlendingNum(const Scene *scene) {



    if (m_config.m_nJacobiIters == 0) return;
    int chunk_size = scene->getBlockSize();
    chunk_size *= chunk_size;

    size_t nNodes = 0;
    for (int i = 0; i < m_preCacheInfoList->size(); i++) {
        auto& pci = (*m_preCacheInfoList)[i];
        PathNode& pn = pci.nodes.front();
        if (!(*m_preCacheInfoList)[i].nodes.front().its.isValid()) {
            pn.maxBlendingNum = 1;
            continue;
        }
        pn.graph_index = nNodes++;
    }

    std::vector<int> components(nNodes);

    typedef std::pair<int, int> Edge;
    std::vector<Edge> edges;
    int nChunks = 1 + ((m_preCacheInfoList->size() - 1) / chunk_size);
    std::vector<int> start_indices(nChunks + 1, 0);

#if defined(MTS_OPENMP)
#pragma omp parallel for schedule(dynamic)
#endif
    for (int cid = 0; cid < nChunks; cid++) {
        for (int i = cid * chunk_size; i < (1 + cid) * chunk_size && i < m_preCacheInfoList->size(); i++) {
            if (!(*m_preCacheInfoList)[i].nodes.front().its.isValid()) continue;
            PathNode& pn = (*m_preCacheInfoList)[i].nodes.front();
            start_indices[cid + 1] += pn.neighbors.size();
        }
    }
    
    for (int cid = 1; cid <= nChunks; cid++) {
        start_indices[cid] += start_indices[cid - 1];
    }

    edges.resize(start_indices[nChunks]);

#if defined(MTS_OPENMP)
#pragma omp parallel for schedule(dynamic)
#endif
    for (int cid = 0; cid < nChunks; cid++) {
        int nEdges = 0;
        int& si = start_indices[cid];
        for (int i = cid * chunk_size; i < (1 + cid) * chunk_size && i < m_preCacheInfoList->size(); i++) {
            if (!(*m_preCacheInfoList)[i].nodes.front().its.isValid()) continue;
            PathNode& pn = (*m_preCacheInfoList)[i].nodes.front();
            for (const auto& neighbor : pn.neighbors) {
                edges[si + (nEdges++)] = std::make_pair(pn.graph_index, neighbor.node->graph_index);
            }
        }
    }

    typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS,
            boost::no_property, boost::no_property, boost::no_property,
            boost::vecS> Graph;

    Graph graph(edges.data(), edges.data() + edges.size(), nNodes);

    int nComp = connected_components(graph, &components[0]);
    std::vector<int> component_sizes(nComp, 0);

#if defined(MTS_OPENMP)
#pragma omp parallel for schedule(dynamic, chunk_size)
#endif
    for (int i = 0; i < components.size(); i++)
        component_sizes[components[i]]++;

#if defined(MTS_OPENMP)
#pragma omp parallel for schedule(dynamic, chunk_size)
#endif
    for (int i = 0; i < m_preCacheInfoList->size(); i++) {
        if (!(*m_preCacheInfoList)[i].nodes.front().its.isValid()) continue;
        PathNode& pn = (*m_preCacheInfoList)[i].nodes.front();
        pn.maxBlendingNum = component_sizes[components[pn.graph_index]];
    }

    // for(int i=0; i<nComp; i++) printf("Size of comp#%d = %d\n", i, component_sizes[i]);

}
#endif

void AdaptiveGradientPathIntegrator::traceDiff(const Scene *scene, const Sensor *sensor, Sampler * sampler) {
    if (m_cancelled) return;
    const int& cx = sensor->getFilm()->getCropSize().x;
    const int& cy = sensor->getFilm()->getCropSize().y;
    const int& bSize = scene->getBlockSize();
    const int& bx = ceil(cx, bSize);
    const int& by = ceil(cy, bSize);

#if defined(MTS_OPENMP)    
    ref<Scheduler> sched = Scheduler::getInstance();
    const int& nCores = sched->getCoreCount();
    ref_vector<Sampler> samplers(nCores);
    for (int i = 0; i < nCores; i++)
        samplers[i] = sampler->clone();
#pragma omp parallel for schedule(dynamic)
#endif    
    for (int blockIndex = 0; blockIndex < bx * by; blockIndex++) {
#if defined(MTS_OPENMP)
        Sampler* sampler = samplers[omp_get_thread_num()];
#endif  
        // Original code from SamplingIntegrator.
        Float diffScaleFactor = 1.0f / std::sqrt((Float) sampler->getSampleCount());
        RadianceQueryRecord rRec(scene, sampler);

        Point2i offset((blockIndex % bx) * bSize, (blockIndex / bx) * bSize);
        sampler->generate(offset);

        for (int pointIndex = 0; pointIndex < bSize * bSize; pointIndex++) {
            int x = offset.x + pointIndex % bSize;
            int y = offset.y + pointIndex / bSize;

            if (x >= cx || y >= cy) continue;

            PrecursorCacheInfo &pci = (*m_preCacheInfoList)[y * cx + x];

            for (int i = 0; i < pci.nodes.front().getSamplingRate(); i++) {
                rRec.newQuery(RadianceQueryRecord::ESensorRay, sensor->getMedium());
                MainRayState mainRay;
                mainRay.throughput = sensor->sampleRayDifferential(mainRay.ray,
                        pci.samplePos, pci.apertureSample, pci.timeSample);
                mainRay.ray.scaleDifferential(diffScaleFactor);
                mainRay.rRec = rRec;
                mainRay.rRec.its = rRec.its;
                mainRay.pci = &pci;
                evaluateDiff(mainRay);
            }

#if defined(USE_RECON_RAYS)
            if (m_currentMode == RECON_MODE) // evaluate diff in the other direction for reconstruction
            {
                PrecursorCacheInfo &pci = (*m_preCacheInfoListBuffer)[y * cx + x];

                rRec.newQuery(RadianceQueryRecord::ESensorRay, sensor->getMedium());

                // Initialize the base path.

                MainRayState mainRay;
                mainRay.accumulateRadiance = false;
                mainRay.throughput = sensor->sampleRayDifferential(mainRay.ray,
                        pci.samplePos, pci.apertureSample, pci.timeSample);
                mainRay.ray.scaleDifferential(diffScaleFactor);
                mainRay.rRec = rRec;
                mainRay.rRec.its = rRec.its;
                mainRay.pci = &pci;
                evaluateDiff(mainRay);
            }
#endif
        }
    }
}

void AdaptiveGradientPathIntegrator::communicateBidirectionalDiff(const Scene * scene, int depth) {
    if (m_cancelled) return;
    int chunk_size = scene->getBlockSize();
    chunk_size *= chunk_size;

    Float avg_dist = 0;
    size_t n_pairs = 0;

    // Do some stats
#if defined(MTS_OPENMP)
#pragma omp parallel for schedule(dynamic, chunk_size) reduction(+: avg_dist, n_pairs)
#endif
    for (int i = 0; i < (*m_preCacheInfoList).size(); i++) {
        auto& pci = (*m_preCacheInfoList)[i];
        if (depth >= pci.nodes.size()) continue;
        auto& node = pci.nodes[depth];
        for (auto& neighbor : node.neighbors) {
            avg_dist += distance(node.its.p, neighbor.node->its.p);
            n_pairs++;
        }
    }
    avg_dist /= n_pairs;

#if defined(BACK_PROP_GRAD)
#if defined(MTS_OPENMP)
#pragma omp parallel for schedule(dynamic, chunk_size)
#endif
    for (int i = 0; i < (*m_preCacheInfoList).size(); i++) {
        auto& pci = (*m_preCacheInfoList)[i];
        if (depth >= pci.nodes.size()) continue;
        auto& node = pci.nodes[depth];
        for (auto& neighbor : node.neighbors) {
            if (depth + 1 < pci.nodes.size()) {
                if (!neighbor.merged) {
                    neighbor.grad = neighbor.grad_before_conn + node.estRad - node.direct_lighting;
                }
            }
        }
    }
#endif

#if defined(MTS_OPENMP)
#pragma omp parallel for schedule(dynamic, chunk_size)
#endif
    for (int i = 0; i < (*m_preCacheInfoList).size(); i++) {
        auto& pci = (*m_preCacheInfoList)[i];
        if (depth >= pci.nodes.size()) continue;
        auto& node = pci.nodes[depth];
        for (auto& neighbor : node.neighbors) {
            neighbor.weight = Spectrum(Float(1));
#if defined(USE_ADAPTIVE_WEIGHT)
            Float dist = distance(node.its.p, neighbor.node->its.p);
            neighbor.weight = Spectrum(exp(-dist / avg_dist));
#endif
            if (neighbor.node > &node) // unidirectional update to avoid conflict
            {
                auto& nn = neighbor.node->neighbors;
                auto it = std::find_if(nn.begin(), nn.end(),
                        [&node] (const PathNode::Neighbor & n) {
                            return n.node == &node;
                        });
                neighbor.grad -= it->grad;
                it->grad = -neighbor.grad;
            }

        }
#if defined(USE_RECON_RAYS)
        if (m_currentMode == RECON_MODE) {
            auto& pci = (*m_preCacheInfoListBuffer)[i];
            for (auto& node : pci.nodes) {
                for (auto& neighbor : node.neighbors) {
                    neighbor.weight = Spectrum(Float(1));
#if defined(USE_ADAPTIVE_WEIGHT)
                    Float dist = distance(node.its.p, neighbor.node->its.p);
                    neighbor.weight = Spectrum(exp(-dist / avg_dist));
#endif
                    if (neighbor.node > &node) // unidirectional update to avoid conflict
                    {
                        auto& nn = neighbor.node->neighbors;
                        auto it = std::find_if(nn.begin(), nn.end(),
                                [&node] (const PathNode::Neighbor & n) {
                                    return n.node == &node;
                                });
                        it->grad /= Float(it->sampleCount);
                        neighbor.grad /= Float(neighbor.sampleCount);
                        neighbor.grad -= it->grad;
                        it->grad = -neighbor.grad;
                    }
                }
            }
        }
#endif
    }
}

void AdaptiveGradientPathIntegrator::iterateJacobi(const Scene * scene, const Sensor *sensor) {
    if (m_cancelled) return;
    size_t chunk_size = scene->getBlockSize();
    chunk_size *= chunk_size;

    Float alpha_sqr = m_config.m_reconstructAlpha;
    alpha_sqr *= alpha_sqr;


    for (int i = m_config.m_maxMergeDepth; i >= m_config.m_minMergeDepth; i--) {
#if defined(BACK_PROP_GRAD)
        communicateBidirectionalDiff(scene, i);
#endif
        
#if defined(GDPT_STYLE_1ST_BOUNCE)
        if (i == 0) continue;
#endif

#if defined(CACHE_FRIENDLY_ITERATOR)
        if (i == 0) {
            const int& cx = sensor->getFilm()->getCropSize().x;
            const int& cy = sensor->getFilm()->getCropSize().y;
            // prepare data

            std::vector<Spectrum> color[2];
            color[0].resize(cx * cy, Spectrum(Float(0)));
            color[1].resize(cx * cy, Spectrum(Float(0)));
            std::vector<Spectrum> grad[4];
            for (int k = 0; k < 4; k++)
                grad[k].resize(cx * cy, Spectrum(std::numeric_limits<Float>::infinity()));
            // 0--(-1, 0)  1--(1, 0) 2--(0, -1) 3--(0, 1)
            // prepare data
            auto& pciList = (*m_preCacheInfoList);
#if defined(MTS_OPENMP)
#pragma omp parallel for
#endif
            for (int y = 0; y < cy; y++) {
                for (int x = 0; x < cx; x++) {
                    int index = y * cx + x;
                    PrecursorCacheInfo& pci = pciList[index];
                    color[0][index] = pci.nodes[0].estRad;
                    for (auto& n : pci.nodes[0].neighbors)
                        grad[n.index][index] = n.grad;
                }
            }

            // iterate
            for (int j = 0; j < m_config.m_nJacobiIters; j++) {
                int src = j % 2;
                int dst = 1 - src;
#if defined(MTS_OPENMP)
#pragma omp parallel for
#endif
                for (int y = 0; y < cy; y++) {
                    for (int x = 0; x < cx; x++) {
                        int index = y * cx + x;
                        int w = 1;
                        color[dst][index] = color[src][index];
                        for (int n = 0; n < 4; n++) {
                            const Spectrum &g = grad[n][index];
                            if (g[0] == std::numeric_limits<Float>::infinity()) continue;
                            color[dst][index] += g;
                            switch (n) {
                                case 0:
                                    color[dst][index] += color[src][index - 1];
                                    break;
                                case 1:
                                    color[dst][index] += color[src][index + 1];
                                    break;
                                case 2:
                                    color[dst][index] += color[src][index - cx];
                                    break;
                                case 3:
                                    color[dst][index] += color[src][index + cx];
                                    break;
                            }
                            w++;
                        }
                        color[dst][index] /= Float(w);
                    }
                }
            }
#if defined(MTS_OPENMP)
#pragma omp parallel for
#endif
            for (int y = 0; y < cy; y++) {
                for (int x = 0; x < cx; x++) {
                    int index = y * cx + x;
                    PrecursorCacheInfo& pci = pciList[index];
                    pci.nodes[0].estRad = color[m_config.m_nJacobiIters % 2][index];
                    pci.nodes[0].estRadBuffer[m_config.m_nJacobiIters % 2] = color[m_config.m_nJacobiIters % 2][index];
                }
            }
            continue;
        }


#endif


        for (int j = 0; j < m_config.m_nJacobiIters; j++) {
            //if(i == 1) break; // for debug
#if defined(USE_RECON_RAYS)
            if (j == 1 && i == 1 && m_currentMode == RECON_MODE) break;
#endif
            int src = j % 2;
            int dst = 1 - src;
            //Float avg_diff = 0.f;
            // update current buffer
            auto& pciList = (*m_preCacheInfoList);

#if defined(MTS_OPENMP)
#pragma omp parallel for schedule(dynamic, chunk_size)// reduction(+: avg_diff)
#endif
            for (size_t index = 0; index < pciList.size(); index++) { // this part needs to be improved by cache friendly design
                auto& pci = pciList[index];

                if (i < pci.nodes.size()) {

                    auto& node = pci.nodes[i];
                    Spectrum color(Float(0));
                    color += node.estRadBuffer[src];
                    Float w = 1;
                    for (auto& n : node.neighbors) {
                        color += n.node->estRadBuffer[src] * n.weight;
                        color += n.grad * n.weight;
                        w += 1;
                    }

                    node.estRadBuffer[dst] = color / w; // update
                    // avg_diff += (node.estRad[dst] - node.estRad[src]).max();
                }

            }
            if (j == 10 && i > 0) break; // for debug
        }

        int dstBuffer = m_config.m_nJacobiIters % 2;
#if defined(USE_RECON_RAYS)
        if (i == 1 && m_currentMode == RECON_MODE) dstBuffer = 1;
#endif
        if (i == 0) continue;

#if defined(MTS_OPENMP)
#pragma omp parallel for schedule(dynamic, chunk_size)
#endif
        // propagate to previous depth
        for (size_t index = 0; index < (*m_preCacheInfoList).size(); index++) {
            auto& pci = (*m_preCacheInfoList)[index];

            if (i >= pci.nodes.size()) continue;
            const Spectrum& dl = pci.nodes[i - 1].direct_lighting;
            pci.nodes[i - 1].estRad = pci.nodes[i].estRadBuffer[dstBuffer] *
                    pci.nodes[i].weight_multiplier + dl;
            pci.nodes[i - 1].estRadBuffer[0] = pci.nodes[i - 1].estRadBuffer[1] = pci.nodes[i - 1].estRad;
        }
    }
}

void AdaptiveGradientPathIntegrator::setOutputBuffer(const Scene *scene, Sensor * sensor, int batchSize) {
    if (m_cancelled) return;
    const int& cx = sensor->getFilm()->getCropSize().x;
    const int& cy = sensor->getFilm()->getCropSize().y;
    const int& bSize = scene->getBlockSize();
    const int& bx = ceil(cx, bSize);
    const int& by = ceil(cy, bSize);

    ref<Film> film = sensor->getFilm();

#if defined(MTS_OPENMP)
    ref<Scheduler> sched = Scheduler::getInstance();
    const int& nCores = sched->getCoreCount();
    ref_vector<ImageBlock> blocks(nCores);
    for (int i = 0; i < nCores; i++) {
        blocks[i] = new ImageBlock(Bitmap::ESpectrumAlphaWeight, Vector2i(bSize, bSize),
                film->getReconstructionFilter());
        blocks[i]->setAllowNegativeValues(true);
    }
#pragma omp parallel for
#else
    ref<ImageBlock> block = new ImageBlock(Bitmap::ESpectrumAlphaWeight, Vector2i(bSize, bSize),
            film->getReconstructionFilter());
#endif    
    for (int blockIndex = 0; blockIndex < bx * by; blockIndex++) {
#if defined(MTS_OPENMP)
        ref<ImageBlock> block = blocks[omp_get_thread_num()];
#endif
#if defined(GDPT_STYLE_1ST_BOUNCE)
        for (int buffID = 0; buffID < 5; buffID++) {
            block->setOffset(Point2i((blockIndex % bx) * bSize, (blockIndex / bx) * bSize));
            block->clear();
            for (int pointIndex = 0; pointIndex < bSize * bSize; pointIndex++) {

                int x = block->getOffset().x + pointIndex % bSize;
                int y = block->getOffset().y + pointIndex / bSize;
                if (x >= cx || y >= cy) continue;

                PrecursorCacheInfo &pci = (*m_preCacheInfoList)[y * cx + x];
                Spectrum color(0.f);
                Spectrum dx(0.f);
                Spectrum dy(0.f);
                if (pci.nodes[0].its.isValid()) {
                    color = pci.nodes[0].estRad * pci.nodes[0].weight_multiplier;
                    dx = color;
                    dy = color;
                    for (auto& n : pci.nodes[0].neighbors) {
                        if (n.index == 1)
                            dx = n.grad;
                        if (n.index == 3)
                            dy = n.grad;
                    }
                }
                else {
                    int x_next = x+1;
                    int y_next = y+1;
                    if(x_next < cx) {
                        auto& nodes = (*m_preCacheInfoList)[y * cx + x_next].nodes;
                        if(nodes.size())
                            dx = -nodes.front().estRad * nodes.front().weight_multiplier;
                    }
                    if(y_next < cy) {
                        auto& nodes = (*m_preCacheInfoList)[y_next * cx + x].nodes;
                        if(nodes.size())
                            dy = -nodes.front().estRad * nodes.front().weight_multiplier;
                    }
                }
                
                switch (buffID) {
                    case BUFFER_FINAL:
                        color += pci.very_direct_lighting;
                        break;
                    case BUFFER_DX:
                        color = -dx;
                        break;
                    case BUFFER_DY:
                        color = -dy;
                        break;
                    case BUFFER_VERY_DIRECT:
                        color = pci.very_direct_lighting;
                        break;
                }
                
                block->put(pci.samplePos, color / Float(batchSize) * pci.factor, 1.f);
            }
            film->putMulti(block, buffID);
        }
#else
        block->setOffset(Point2i((blockIndex % bx) * bSize, (blockIndex / bx) * bSize));
        block->clear();
        for (int pointIndex = 0; pointIndex < bSize * bSize; pointIndex++) {

            int x = block->getOffset().x + pointIndex % bSize;
            int y = block->getOffset().y + pointIndex / bSize;
            if (x >= cx || y >= cy) continue;

            PrecursorCacheInfo &pci = (*m_preCacheInfoList)[y * cx + x];
            int bufferID = m_config.m_nJacobiIters % 2;
            if (pci.nodes.size() >= 1) {
                Spectrum color = pci.nodes[0].estRadBuffer[bufferID] * pci.nodes[0].weight_multiplier;
                color += pci.very_direct_lighting;
                block->put(pci.samplePos, color / Float(batchSize) * pci.factor, 1.f);
                //block->put(pci.samplePos, Spectrum(pci.nodes[0].getSamplingRate() / 25.0), 1.f); // output sampling rate
            }
        }
        film->put(block);
#endif
    }
}

bool AdaptiveGradientPathIntegrator::render(Scene *scene,
        RenderQueue *queue, const RenderJob *job,
        int sceneResID, int sensorResID, int samplerResID) {
    m_cancelled = false;

    if (m_hideEmitters) {
        /* Not supported! */
        Log(EError, "Option 'hideEmitters' not implemented for Gradient-Domain Path Tracing!");
    }

    /* Get config from the parent class. */
    m_config.m_maxDepth = m_maxDepth;
    m_config.m_rrDepth = m_rrDepth;
    m_config.m_strictNormals = m_strictNormals;

    /* Code duplicated from SamplingIntegrator::Render. */
    ref<Scheduler> sched = Scheduler::getInstance();
    ref<Sensor> sensor = static_cast<Sensor *> (sched->getResource(sensorResID));

    /* Set up MultiFilm. */
    ref<Film> film = sensor->getFilm();

#if defined(GDPT_STYLE_1ST_BOUNCE)
    std::vector<std::string> outNames;
    outNames.push_back("-final");
    outNames.push_back("-throughput");
    outNames.push_back("-dx");
    outNames.push_back("-dy");
    outNames.push_back("-direct");
    if (!film->setBuffers(outNames)) {
        Log(EError, "Cannot render image! G-PT has been called without MultiFilm.");
        return false;
    }
#endif

    size_t nCores = sched->getCoreCount();
    Sampler *sampler = static_cast<Sampler *> (sched->getResource(samplerResID, 0));
    size_t sampleCount = sampler->getSampleCount();

    Log(EInfo, "Starting render job (GPT::render) (%ix%i, " SIZE_T_FMT " %s, " SIZE_T_FMT
            " %s, " SSE_STR ") ..", film->getCropSize().x, film->getCropSize().y,
            sampleCount, sampleCount == 1 ? "sample" : "samples", nCores,
            nCores == 1 ? "core" : "cores");
    /* This is a sampling-based integrator - parallelize. */
    bool success = true;

    m_process = NULL;

    const int& cx = sensor->getFilm()->getCropSize().x;
    const int& cy = sensor->getFilm()->getCropSize().y;
    m_preCacheInfoList.reset(new std::vector<PrecursorCacheInfo>());
    (*m_preCacheInfoList).resize(cx * cy);

    m_pc.reset(new PointCloud);

#if defined(USE_RECON_RAYS)
    const int n_swap_iters = 8;
    m_pcBuffer.reset(new PointCloud);
    m_preCacheInfoListBuffer.reset(new std::vector<PrecursorCacheInfo>());
    (*m_preCacheInfoListBuffer).resize(cx * cy);
#endif

#if defined(MTS_OPENMP)
    Thread::initializeOpenMP(nCores);
#endif

    m_precursorTask = PRECURSOR_LOOP;

    for (int i = 0; i < sampler->getSampleCount(); i += m_config.m_batchSize) {

#if defined(USE_RECON_RAYS)
        m_currentMode = i % n_swap_iters == 0 ? SAMPLE_MODE : RECON_MODE;
#endif

        // trace precursor
#if defined(PRINT_TIMING)
        MyTimer totalTimer, timer;
        totalTimer.tic();
        timer.tic();
        printf("Iteration #%d:\n", i);
#endif
        tracePrecursor(scene, sensor, sampler);
#if defined(PRINT_TIMING)
        printf("%-20s %5.0lf ms\n", "precursor", timer.toc()*1e3);
#endif

        // figure out neighbors
#if defined(PRINT_TIMING)
        timer.tic();
#endif       
        decideNeighbors(scene, sensor);
#if defined(PRINT_TIMING)
        printf("%-20s %5.0lf ms\n", "neighbors", timer.toc()*1e3);
#endif

#if defined(ADAPTIVE_GRAPH_SAMPLING)
        // figure out blending number for each vertex through graph connectivity
#if defined(PRINT_TIMING)
        timer.tic();
#endif
        getMaxBlendingNum(scene);
#if defined(PRINT_TIMING)
        printf("%-20s %5.0lf ms\n", "get-blending-num", timer.toc()*1e3);
#endif
#endif

        int batchSize;
        for (batchSize = 0; batchSize < m_config.m_batchSize; batchSize++) {
            if (i + batchSize >= sampler->getSampleCount()) break;
            // trace difference
#if defined(PRINT_TIMING)
            timer.tic();
#endif          
            traceDiff(scene, sensor, sampler);
#if defined(PRINT_TIMING)
            printf("%-20s %5.0lf ms\n", "difference", timer.toc()*1e3);
#endif
        }
        // merge bidirectional difference samples

#if !defined(BACK_PROP_GRAD)
#if defined(PRINT_TIMING)
        timer.tic();
#endif        
        communicateBidirectionalDiff(scene);
#if defined(PRINT_TIMING)
        printf("%-20s %5.0lf ms\n", "bidir-merging", timer.toc()*1e3);
#endif
#endif

        // Jacobi iterations
#if defined(PRINT_TIMING)
        timer.tic();
#endif          
        iterateJacobi(scene, sensor);
#if defined(PRINT_TIMING)
        printf("%-20s %5.0lf ms\n", "Jacobi-iteration", timer.toc()*1e3);
#endif
        // output
        setOutputBuffer(scene, sensor, batchSize);

#if defined(PRINT_TIMING)
        printf("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n");
        printf("%-20s %5.0lf ms\n", "total", totalTimer.toc()*1e3);
        printf("\n");
#endif
        if (m_cancelled) {
            queue->signalFinishJob(job, true);
            break;
        }

#if defined(USE_RECON_RAYS)
        // swap buffer to get fresh indirect lighting samples
        if (m_currentMode == SAMPLE_MODE)
            m_preCacheInfoList.swap(m_preCacheInfoListBuffer);
#endif

        queue->signalRefresh(job);
    }

#if defined(GDPT_STYLE_1ST_BOUNCE)
    ref<Bitmap> throughputBitmap(new Bitmap(Bitmap::ESpectrum, Bitmap::EFloat, film->getCropSize()));
    ref<Bitmap> directBitmap(new Bitmap(Bitmap::ESpectrum, Bitmap::EFloat, film->getCropSize()));
    ref<Bitmap> dxBitmap(new Bitmap(Bitmap::ESpectrum, Bitmap::EFloat, film->getCropSize()));
    ref<Bitmap> dyBitmap(new Bitmap(Bitmap::ESpectrum, Bitmap::EFloat, film->getCropSize()));
    ref<Bitmap> reconstructionBitmap(new Bitmap(Bitmap::ESpectrum, Bitmap::EFloat, film->getCropSize()));

    film->developMulti(Point2i(0, 0), film->getCropSize(), Point2i(0, 0), throughputBitmap, BUFFER_THROUGHPUT);
    film->developMulti(Point2i(0, 0), film->getCropSize(), Point2i(0, 0), dxBitmap, BUFFER_DX);
    film->developMulti(Point2i(0, 0), film->getCropSize(), Point2i(0, 0), dyBitmap, BUFFER_DY);
    film->developMulti(Point2i(0, 0), film->getCropSize(), Point2i(0, 0), directBitmap, BUFFER_VERY_DIRECT);

    /* Transform the data for the solver. */
    int w = film->getCropSize().x;
    int h = film->getCropSize().y;
    size_t subPixelCount = 3 * w * h;
    std::vector<float> throughputVector(subPixelCount);
    std::vector<float> dxVector(subPixelCount);
    std::vector<float> dyVector(subPixelCount);
    std::vector<float> directVector(subPixelCount);
    std::vector<float> reconstructionVector[2];
    reconstructionVector[0].resize(subPixelCount);
    reconstructionVector[1].resize(subPixelCount);

    std::transform(throughputBitmap->getFloatData(), throughputBitmap->getFloatData() + subPixelCount, throughputVector.begin(), [](Float x) {
        return (float) x; });
    std::transform(throughputBitmap->getFloatData(), throughputBitmap->getFloatData() + subPixelCount, reconstructionVector[0].begin(), [](Float x) {
        return (float) x; });
    std::transform(dxBitmap->getFloatData(), dxBitmap->getFloatData() + subPixelCount, dxVector.begin(), [](Float x) {
        return (float) x; });
    std::transform(dyBitmap->getFloatData(), dyBitmap->getFloatData() + subPixelCount, dyVector.begin(), [](Float x) {
        return (float) x; });
    std::transform(directBitmap->getFloatData(), directBitmap->getFloatData() + subPixelCount, directVector.begin(), [](Float x) {
        return (float) x; });
        
    int chunk_size = scene->getBlockSize();
    chunk_size *= chunk_size;

    const int& n_iters = m_config.m_nJacobiIters;
    for (int iter = 0; iter < n_iters; iter++) {
        int src = iter % 2;
        int dst = 1 - src;
        
#if defined(MTS_OPENMP)
#pragma omp parallel for schedule(dynamic, chunk_size)
#endif
        for (int i = 0; i < w * h; i++) {
            int x = i % w;
            int y = i / w;
            if (x >= w || y >= h) continue;
            Vector3f color(0.f);
            float weight = 0.f;
            const Vector3f& prim = ((Vector3f*) reconstructionVector[src].data())[y * w + x];
            color += prim;
            weight += 1;
            if (x > 0) {
                color += ((Vector3f*) dxVector.data())[y * w + x - 1];
                color += ((Vector3f*) reconstructionVector[src].data())[y * w + x - 1];
                weight += 1.f;
            }
            if (x + 1 < w) {
                color -= ((Vector3f*) dxVector.data())[y * w + x];
                color += ((Vector3f*) reconstructionVector[src].data())[y * w + x + 1];
                weight += 1.f;
            }
            if (y > 0) {
                color += ((Vector3f*) dyVector.data())[(y - 1) * w + x];
                color += ((Vector3f*) reconstructionVector[src].data())[(y - 1) * w + x];
                weight += 1.f;
            }
            if (y + 1 < h) {
                color -= ((Vector3f*) dyVector.data())[y * w + x];
                color += ((Vector3f*) reconstructionVector[src].data())[(y + 1) * w + x];
                weight += 1.f;
            }
            ((Vector3f*) reconstructionVector[dst].data())[y * w + x] = color / weight;

        }
    }


#if defined(MTS_OPENMP)
#pragma omp parallel for schedule(dynamic, chunk_size)
#endif
    for (int i = 0; i < w * h; i++) {
        int x = i % w;
        int y = i / w;
        if (x >= w || y >= h) continue;
        const Vector3f& color = ((Vector3f*) reconstructionVector[n_iters % 2].data())[y * w + x];
        const Vector3f& direct = ((Vector3f*) directVector.data())[y * w + x];
        Float specColor[] = {color.x, color.y, color.z};
        Float specDirect[] = {direct.x, direct.y, direct.z};
        reconstructionBitmap->setPixel(Point2i(x, y), Spectrum(specColor) + Spectrum(specDirect));
    }
    film->setBitmapMulti(reconstructionBitmap, 1, BUFFER_FINAL);
    queue->signalRefresh(job);
#endif

    return success;
}

static StatsCounter avgPathLength("Unstructured Gradient Path Tracer", "Average path length", EAverage);

void AdaptiveGradientPathIntegrator::evaluatePrecursor(MainRayState & main) {
    const Scene *scene = main.rRec.scene;
    // Perform the first ray intersection for the base path (or ignore if the intersection has already been provided).

    main.pci->nodes.push_back(PathNode());
    main.pci->nodes.back().lastRay = main.ray;
    main.rRec.rayIntersect(main.ray);
    main.pci->nodes.back().its = main.rRec.its; // add cache info
    main.pci->nodes.back().weight_multiplier = main.throughput / main.pdf;
    main.ray.mint = Epsilon;

    if (!main.rRec.its.isValid()) return;

    // Strict normals check to produce the same results as bidirectional methods when normal mapping is used.
    if (m_config.m_strictNormals) {
        // If 'strictNormals'=true, when the geometric and shading normals classify the incident direction to the same side, then the main path is still good.
        if (dot(main.ray.d, main.rRec.its.geoFrame.n) * Frame::cosTheta(main.rRec.its.wi) >= 0) {
            // This is an impossible base path.
            return;
        }
    }

    // Main path tracing loop.
    main.rRec.depth = 1;

    while (main.rRec.depth < m_config.m_maxDepth || m_config.m_maxDepth < 0) {
        // Strict normals check to produce the same results as bidirectional methods when normal mapping is used.
        // If 'strictNormals'=true, when the geometric and shading normals classify the incident direction to the same side, then the main path is still good.
        if (m_config.m_strictNormals) {
            if (dot(main.ray.d, main.rRec.its.geoFrame.n) * Frame::cosTheta(main.rRec.its.wi) >= 0) {
                // This is an impossible main path, and there are no more paths to shift.
                return;
            }
        }

        main.pci->nodes.back().vertexType = getVertexType(main, m_config, BSDF::ESmooth);

        // break if enough number of nodes has been collected
        if (main.pci->nodes.size() > m_config.m_maxMergeDepth) break;

        Point2 sample = main.rRec.nextSample2D();
        Point3 full_sample;
        full_sample.x = sample.x;
        full_sample.y = sample.y;
        full_sample.z = main.rRec.nextSample1D();
        main.pci->nodes.back().bsdfSample = full_sample; // cache bsdf sample
        // Sample a new direction from BSDF * cos(theta).

        
        BSDFSampleResult mainBsdfResult = sampleBSDF(main, full_sample);


        if (mainBsdfResult.pdf <= (Float) 0.0) {
            // Impossible base path.
            break;
        }

        const Vector mainWo = main.rRec.its.toWorld(mainBsdfResult.bRec.wo);
        
        main.pci->nodes.back().mainWo = mainWo; // for debug

        // Prevent light leaks due to the use of shading normals.
        Float mainWoDotGeoN = dot(main.rRec.its.geoFrame.n, mainWo);
        if (m_config.m_strictNormals && mainWoDotGeoN * Frame::cosTheta(mainBsdfResult.bRec.wo) <= 0) {
            break;
        }



        main.ray = Ray(main.rRec.its.p, mainWo, main.ray.time);
        Ray prevRay = main.ray;

        scene->rayIntersect(main.ray, main.rRec.its);
        // ignore specular nodes
        //if(main.pci->nodes.back().vertexType == VERTEX_TYPE_DIFFUSE)
        main.pci->nodes.push_back(PathNode());
        main.pci->nodes.back().its = main.rRec.its; // add cache info
        main.pci->nodes.back().lastRay = prevRay;
        main.pci->nodes.back().weight_multiplier = mainBsdfResult.weight;
        if (!main.rRec.its.isValid()) break;

        main.throughput *= mainBsdfResult.weight * mainBsdfResult.pdf;
        main.pdf *= mainBsdfResult.pdf;
        main.eta *= mainBsdfResult.bRec.eta;

        // Stop if the base path hit the environment.
        main.rRec.type = RadianceQueryRecord::ERadianceNoEmission;
        if (!(main.rRec.type & RadianceQueryRecord::EIndirectSurfaceRadiance)) {
            break;
        }


        if (main.rRec.depth++ >= m_config.m_rrDepth) {
            /* Russian roulette: try to keep path weights equal to one,
               while accounting for the solid angle compression at refractive
               index boundaries. Stop with at least some probability to avoid
               getting stuck (e.g. due to total internal reflection) */
            Float q = std::min((main.throughput / (D_EPSILON + main.pdf)).max() * main.eta * main.eta, (Float) 0.95f);

            main.pci->nodes.back().rrSample = main.rRec.nextSample1D();
            if (main.pci->nodes.back().rrSample >= q) {
                main.pci->nodes.back().bsdfSample = Point3(-1.f, -1.f, -1.f);
                break;
            }

            main.pdf *= q;
        }
    }
    for (int i = 1; i < main.pci->nodes.size(); i++) {
        main.pci->nodes[i].current_weight = main.pci->nodes[i - 1].current_weight * main.pci->nodes[i].weight_multiplier;
    }
}

void AdaptiveGradientPathIntegrator::evaluateDiff(MainRayState& main, BranchArguments* branchArguments) { // evaluate difference to neighbors
    const Scene *scene = main.rRec.scene;

    std::vector<ShiftedRayState> shiftedRays;

    if (!branchArguments) {
        shiftedRays.reserve(20);

        main.rRec.depth = 0;

        if (main.pci->nodes.size()) {
            main.rRec.its = main.pci->nodes[0].its; // get cached intersection
        } else
            main.rRec.rayIntersect(main.ray);

        main.ray.mint = Epsilon;

        Spectrum & out_veryDirect = main.pci->very_direct_lighting;
        if (!main.rRec.its.isValid()) {
            // First hit is not in the scene so can't continue. Also there there are no paths to shift.

            // Add potential very direct light from the environment as gradients are not used for that.
            if (main.rRec.type & RadianceQueryRecord::EEmittedRadiance) {
                out_veryDirect += main.throughput * scene->evalEnvironment(main.ray);
            }

            //SLog(EInfo, "Main ray(%d): First hit not in scene.", rayCount);
            return;
        }
        // Add very direct light from non-environment.
        {
            // Include emitted radiance if requested.
            if (main.rRec.its.isEmitter() && (main.rRec.type & RadianceQueryRecord::EEmittedRadiance)) {
                out_veryDirect += main.throughput * main.rRec.its.Le(-main.ray.d);
            }

            // Include radiance from a subsurface scattering model if requested. Note: Not tested!
            if (main.rRec.its.hasSubsurface() && (main.rRec.type & RadianceQueryRecord::ESubsurfaceRadiance)) {
                out_veryDirect += main.throughput * main.rRec.its.LoSub(scene, main.rRec.sampler, -main.ray.d, 0);
            }
        }

        // Main path tracing loop.
        main.rRec.depth = 1;
    } else {
        shiftedRays = branchArguments->shiftedRays;
    }



    while (main.rRec.depth < m_config.m_maxDepth || m_config.m_maxDepth < 0) {
        if (!branchArguments) {
            main.spawnShiftedRay(shiftedRays); // spawn shifted rays for current depth
        }
#if defined(ADAPTIVE_DIFF_SAMPLING)
        if (main.rRec.depth == 2 && !branchArguments) {

            std::vector<int> splitNum(shiftedRays.size(), 0);
            for (int i = 0; i < shiftedRays.size(); i++) {
                auto& shifted = shiftedRays[i];
                if (shifted.activeDepth > 2) continue;

                Float importance = shifted.getImportance();
                if (shifted.alive)
                    splitNum[i] = std::floor(importance * importance * 8);

                shifted.neighbor->sampleCount = 1 + splitNum[i];
            }

            while (1) {
                BranchArguments ba;
                std::vector<ShiftedRayState> &splitRays = ba.shiftedRays;
                bool done = true;
                splitRays.clear();
                for (int i = 0; i < shiftedRays.size(); i++) {
                    if (splitNum[i] == 0) continue;
                    done = false;
                    splitNum[i]--;
                    splitRays.push_back(shiftedRays[i]);
                }
                if (done) break;
                MainRayState duplicateMain = main;
                duplicateMain.rRec = main.rRec;
                duplicateMain.rRec.its = main.rRec.its;
                evaluateDiff(duplicateMain, &ba); // trace branched rays
            }
        }
#endif



        // Strict normals check to produce the same results as bidirectional methods when normal mapping is used.
        // If 'strictNormals'=true, when the geometric and shading normals classify the incident direction to the same side, then the main path is still good.
        if (m_config.m_strictNormals) {
            if (dot(main.ray.d, main.rRec.its.geoFrame.n) * Frame::cosTheta(main.rRec.its.wi) >= 0) {
                // This is an impossible main path, and there are no more paths to shift.
                return;
            }

            for (auto& shifted : shiftedRays) {
                if (dot(shifted.ray.d, shifted.its.geoFrame.n) * Frame::cosTheta(shifted.its.wi) >= 0) {
                    // This is an impossible offset path.
                    shifted.alive = false;
                }
            }
        }

        // Some optimizations can be made if this is the last traced segment.
        bool lastSegment = (main.rRec.depth + 1 == m_config.m_maxDepth);

        /* ==================================================================== */
        /*                     Direct illumination sampling                     */
        /* ==================================================================== */


        // Sample incoming radiance from lights (next event estimation).
        {
            const BSDF* mainBSDF = main.rRec.its.getBSDF(main.ray);

            if (main.rRec.type & RadianceQueryRecord::EDirectSurfaceRadiance && mainBSDF->getType() & BSDF::ESmooth && main.rRec.depth + 1 >= m_config.m_minDepth) {
                // Sample an emitter and evaluate f = f/p * p for it. */
                DirectSamplingRecord dRec(main.rRec.its);

                mitsuba::Point2 lightSample = main.rRec.nextSample2D();

                std::pair<Spectrum, bool> emitterTuple = scene->sampleEmitterDirectVisible(dRec, lightSample);
                Spectrum mainEmitterRadiance = emitterTuple.first * dRec.pdf;
                bool mainEmitterVisible = emitterTuple.second;

                const Emitter *emitter = static_cast<const Emitter *> (dRec.object);

                // If the emitter sampler produces a non-emitter, that's a problem.
                SAssert(emitter != NULL);

                // Add radiance and gradients to the base path and its offset path.
                // Query the BSDF to the emitter's direction.

                BSDFSamplingRecord mainBRec(main.rRec.its, main.rRec.its.toLocal(dRec.d), ERadiance);


                // Evaluate BSDF * cos(theta).
                Spectrum mainBSDFValue = mainBSDF->eval(mainBRec);

                // Calculate the probability density of having generated the sampled path segment by BSDF sampling. Note that if the emitter is not visible, the probability density is zero.
                // Even if the BSDF sampler has zero probability density, the light sampler can still sample it.
                Float mainBsdfPdf = (emitter->isOnSurface() && dRec.measure == ESolidAngle && mainEmitterVisible) ? mainBSDF->pdf(mainBRec) : 0;

#if defined(FACTOR_MATERIAL)
                if (main.rRec.depth == 1 && mainBSDFValue.max() > Float(0)) // for test
                {
                    mainBSDFValue /= Spectrum(D_EPSILON) + mainBSDF->getDiffuseReflectance(main.rRec.its);
                }
#endif

                // There values are probably needed soon for the Jacobians.
                Float mainDistanceSquared = (main.rRec.its.p - dRec.p).lengthSquared();
                Float mainOpposingCosine = dot(dRec.n, (main.rRec.its.p - dRec.p)) / sqrt(mainDistanceSquared);

                // Power heuristic weights for the following strategies: light sample from base, BSDF sample from base.
                Float mainWeightNumerator = main.pdf * dRec.pdf;
                Float mainWeightDenominator = (main.pdf * main.pdf) * ((dRec.pdf * dRec.pdf) + (mainBsdfPdf * mainBsdfPdf));

                Spectrum estimated_radiance = Spectrum(main.pdf * mainWeightNumerator / (D_EPSILON + mainWeightDenominator));
                estimated_radiance *= (mainBSDFValue * mainEmitterRadiance);

                if (estimated_radiance.max() > Float(0) && !branchArguments)
                    main.addRadiance(estimated_radiance);


                // Strict normals check to produce the same results as bidirectional methods when normal mapping is used.
                if (!m_config.m_strictNormals || dot(main.rRec.its.geoFrame.n, dRec.d) * Frame::cosTheta(mainBRec.wo) > 0) {
                    // The base path is good. Add radiance differences to offset paths.
                    for (auto& shifted : shiftedRays) {

                        Spectrum &main_throughput = shifted.main_throughput;
                        Float &main_pdf = shifted.main_pdf;
                        Float mainWeightNumerator = main_pdf * dRec.pdf;
                        Float mainWeightDenominator = (main_pdf * main_pdf) * ((dRec.pdf * dRec.pdf) + (mainBsdfPdf * mainBsdfPdf));

                        Spectrum mainContribution(Float(0));
                        Spectrum shiftedContribution(Float(0));
                        Float weight = Float(0);

                        bool shiftSuccessful = shifted.alive;

                        // Construct the offset path.
                        if (shiftSuccessful) {
                            // Generate the offset path.
                            if (shifted.connection_status == RAY_CONNECTED) {
                                // Follow the base path. All relevant vertices are shared. 
                                Float shiftedBsdfPdf = mainBsdfPdf;
                                Float shiftedDRecPdf = dRec.pdf;
                                Spectrum shiftedBsdfValue = mainBSDFValue;
                                Spectrum shiftedEmitterRadiance = mainEmitterRadiance;
                                Float jacobian = (Float) 1;

                                // Power heuristic between light sample from base, BSDF sample from base, light sample from offset, BSDF sample from offset.
                                Float shiftedWeightDenominator = (jacobian * shifted.pdf) * (jacobian * shifted.pdf) * ((shiftedDRecPdf * shiftedDRecPdf) + (shiftedBsdfPdf * shiftedBsdfPdf));
                                weight = mainWeightNumerator / (D_EPSILON + shiftedWeightDenominator + mainWeightDenominator);


                                mainContribution = main_throughput * (mainBSDFValue * mainEmitterRadiance);
                                shiftedContribution = jacobian * shifted.throughput * (shiftedBsdfValue * shiftedEmitterRadiance);

                                // Note: The Jacobians were baked into shifted.pdf and shifted.throughput at connection phase.
                            } else if (shifted.connection_status == RAY_RECENTLY_CONNECTED) {
                                // Follow the base path. The current vertex is shared, but the incoming directions differ.
                                Vector3 incomingDirection = normalize(shifted.its.p - main.rRec.its.p);

                                BSDFSamplingRecord bRec(main.rRec.its, main.rRec.its.toLocal(incomingDirection), main.rRec.its.toLocal(dRec.d), ERadiance);

                                // Sample the BSDF.
                                Float shiftedBsdfPdf = (emitter->isOnSurface() && dRec.measure == ESolidAngle && mainEmitterVisible) ? mainBSDF->pdf(bRec) : 0; // The BSDF sampler can not sample occluded path segments.
                                Float shiftedDRecPdf = dRec.pdf;
                                Spectrum shiftedBsdfValue = mainBSDF->eval(bRec);

#if defined(FACTOR_MATERIAL)
                                if (main.rRec.depth == 1 && shiftedBsdfValue.max() > Float(0)) // for test
                                {
                                    shiftedBsdfValue /= Spectrum(D_EPSILON) + mainBSDF->getDiffuseReflectance(main.rRec.its);
                                }
#endif

                                Spectrum shiftedEmitterRadiance = mainEmitterRadiance;
                                Float jacobian = (Float) 1;

                                // Power heuristic between light sample from base, BSDF sample from base, light sample from offset, BSDF sample from offset.
                                Float shiftedWeightDenominator = (jacobian * shifted.pdf) * (jacobian * shifted.pdf) * ((shiftedDRecPdf * shiftedDRecPdf) + (shiftedBsdfPdf * shiftedBsdfPdf));
                                weight = mainWeightNumerator / (D_EPSILON + shiftedWeightDenominator + mainWeightDenominator);


                                mainContribution = main_throughput * (mainBSDFValue * mainEmitterRadiance);
                                shiftedContribution = jacobian * shifted.throughput * (shiftedBsdfValue * shiftedEmitterRadiance);

                                // Note: The Jacobians were baked into shifted.pdf and shifted.throughput at connection phase.
                            } else {
                                // Reconnect to the sampled light vertex. No shared vertices.
                                SAssert(shifted.connection_status == RAY_NOT_CONNECTED);

                                const BSDF* shiftedBSDF = shifted.its.getBSDF(shifted.ray);

                                // This implementation uses light sampling only for the reconnect-shift.
                                // When one of the BSDFs is very glossy, light sampling essentially reduces to a failed shift anyway.
                                bool mainAtPointLight = (dRec.measure == EDiscrete);

                                VertexType mainVertexType = getVertexType(main, m_config, BSDF::ESmooth);
                                VertexType shiftedVertexType = getVertexType(shifted, m_config, BSDF::ESmooth);

                                if (mainAtPointLight || (mainVertexType == VERTEX_TYPE_DIFFUSE && shiftedVertexType == VERTEX_TYPE_DIFFUSE)) {
                                    // Get emitter radiance.
                                    DirectSamplingRecord shiftedDRec(shifted.its);
                                    std::pair<Spectrum, bool> emitterTuple = scene->sampleEmitterDirectVisible(shiftedDRec, lightSample);
                                    bool shiftedEmitterVisible = emitterTuple.second;

                                    Spectrum shiftedEmitterRadiance = emitterTuple.first * shiftedDRec.pdf;
                                    Float shiftedDRecPdf = shiftedDRec.pdf;

                                    // Sample the BSDF.
                                    Float shiftedDistanceSquared = (dRec.p - shifted.its.p).lengthSquared();
                                    Vector emitterDirection = (dRec.p - shifted.its.p) / sqrt(shiftedDistanceSquared);
                                    Float shiftedOpposingCosine = -dot(dRec.n, emitterDirection);

                                    BSDFSamplingRecord bRec(shifted.its, shifted.its.toLocal(emitterDirection), ERadiance);

                                    // Strict normals check, to make the output match with bidirectional methods when normal maps are present.
                                    if (m_config.m_strictNormals && dot(shifted.its.geoFrame.n, emitterDirection) * Frame::cosTheta(bRec.wo) < 0) {
                                        // Invalid, non-samplable offset path.
                                        shiftSuccessful = false;
                                    } else {
                                        Spectrum shiftedBsdfValue = shiftedBSDF->eval(bRec);
                                        Float shiftedBsdfPdf = (emitter->isOnSurface() && dRec.measure == ESolidAngle && shiftedEmitterVisible) ? shiftedBSDF->pdf(bRec) : 0;
#if defined(FACTOR_MATERIAL)
                                        if (main.rRec.depth == 1 && shiftedBsdfValue.max() > Float(0)) // for test
                                        {
                                            shiftedBsdfValue /= Spectrum(D_EPSILON) + shiftedBSDF->getDiffuseReflectance(shifted.its);
                                        }
#endif
                                        Float jacobian = std::abs(shiftedOpposingCosine * mainDistanceSquared) / (Epsilon + std::abs(mainOpposingCosine * shiftedDistanceSquared));

                                        // Power heuristic between light sample from base, BSDF sample from base, light sample from offset, BSDF sample from offset.
                                        Float shiftedWeightDenominator = (jacobian * shifted.pdf) * (jacobian * shifted.pdf) * ((shiftedDRecPdf * shiftedDRecPdf) + (shiftedBsdfPdf * shiftedBsdfPdf));
                                        weight = mainWeightNumerator / (D_EPSILON + shiftedWeightDenominator + mainWeightDenominator);

                                        mainContribution = main_throughput * (mainBSDFValue * mainEmitterRadiance);
                                        shiftedContribution = jacobian * shifted.throughput * (shiftedBsdfValue * shiftedEmitterRadiance);
                                    }
                                }
                            }
                        }

                        if (!shiftSuccessful) {
                            // The offset path cannot be generated; Set offset PDF and offset throughput to zero. This is what remains.

                            // Power heuristic between light sample from base, BSDF sample from base, light sample from offset, BSDF sample from offset. (Offset path has zero PDF)
                            weight = mainWeightNumerator / (D_EPSILON + mainWeightDenominator);

                            mainContribution = main_throughput * (mainBSDFValue * mainEmitterRadiance);
                            shiftedContribution = Spectrum((Float) 0);
                        }

                        // Note: Using also the offset paths for the throughput estimate, like we do here, provides some advantage when a large reconstruction alpha is used,
                        // but using only throughputs of the base paths doesn't usually lose by much.
                        shifted.addGradient(mainContribution, shiftedContribution, weight);
                    }
                } // Strict normals
            }
        } // Sample incoming radiance from lights.

        /* ==================================================================== */
        /*               BSDF sampling and emitter hits                         */
        /* ==================================================================== */

        // Sample a new direction from BSDF * cos(theta).
        Point3 full_sample;

        if (main.rRec.depth - 1 < main.pci->nodes.size() && !branchArguments && main.pci->nodes[main.rRec.depth - 1].bsdfSample.x > -Float(0.5)) {
            full_sample = main.pci->nodes[main.rRec.depth - 1].bsdfSample;
        } else {
            Point2 sample = main.rRec.nextSample2D();
            full_sample.x = sample.x;
            full_sample.y = sample.y;
            full_sample.z = main.rRec.nextSample1D();
        }
        

        BSDFSampleResult mainBsdfResult = sampleBSDF(main, full_sample);
#if defined(FACTOR_MATERIAL)
        if (main.rRec.depth == 1) // for test
        {
            main.pci->factor = main.rRec.its.getBSDF()->getDiffuseReflectance(main.rRec.its);
            mainBsdfResult.weight /= Spectrum(D_EPSILON) + main.pci->factor;
        }
#endif

        if (mainBsdfResult.pdf <= (Float) 0.0) {
            // Impossible base path.
            break;
        }

        const Vector mainWo = main.rRec.its.toWorld(mainBsdfResult.bRec.wo);

        // Prevent light leaks due to the use of shading normals.
        Float mainWoDotGeoN = dot(main.rRec.its.geoFrame.n, mainWo);
        if (m_config.m_strictNormals && mainWoDotGeoN * Frame::cosTheta(mainBsdfResult.bRec.wo) <= 0) {
            break;
        }

        // The old intersection structure is still needed after main.rRec.its gets updated.
        Intersection previousMainIts = main.rRec.its;

        // Trace a ray in the sampled direction.
        bool mainHitEmitter = false;
        Spectrum mainEmitterRadiance = Spectrum((Float) 0);

        DirectSamplingRecord mainDRec(main.rRec.its);
        const BSDF* mainBSDF = main.rRec.its.getBSDF(main.ray);


        // Update the vertex types.
        VertexType mainVertexType = getVertexType(main, m_config, mainBsdfResult.bRec.sampledType);
        VertexType mainNextVertexType;

        main.ray = Ray(main.rRec.its.p, mainWo, main.ray.time);

        if (main.rRec.depth < main.pci->nodes.size() && !branchArguments) {
            main.rRec.its = main.pci->nodes[main.rRec.depth].its; // use cached intersection if possible
            main.ray = main.pci->nodes[main.rRec.depth].lastRay;
        } else
            scene->rayIntersect(main.ray, main.rRec.its);

        if (main.rRec.its.isValid()) {
            // Intersected something - check if it was a luminaire.
            if (main.rRec.its.isEmitter()) {
                mainEmitterRadiance = main.rRec.its.Le(-main.ray.d);

                mainDRec.setQuery(main.ray, main.rRec.its);
                mainHitEmitter = true;
            }

            // Sub-surface scattering.
            if (main.rRec.its.hasSubsurface() && (main.rRec.type & RadianceQueryRecord::ESubsurfaceRadiance)) {
                mainEmitterRadiance += main.rRec.its.LoSub(scene, main.rRec.sampler, -main.ray.d, main.rRec.depth);
            }

            // Update the vertex type.
            mainNextVertexType = getVertexType(main, m_config, mainBsdfResult.bRec.sampledType);
        } else {
            // Intersected nothing -- perhaps there is an environment map?
            const Emitter *env = scene->getEnvironmentEmitter();

            if (env) {
                // Hit the environment map.
                mainEmitterRadiance = env->evalEnvironment(main.ray);
                if (!env->fillDirectSamplingRecord(mainDRec, main.ray))
                    break;
                mainHitEmitter = true;

                // Handle environment connection as diffuse (that's ~infinitely far away).

                // Update the vertex type.
                mainNextVertexType = VERTEX_TYPE_DIFFUSE;
            } else {
                // Nothing to do anymore.
                break;
            }
        }

        // Continue the shift.

        // Compute the probability density of generating base path's direction using the implemented direct illumination sampling technique.
        const Float mainLumPdf = (mainHitEmitter && main.rRec.depth + 1 >= m_config.m_minDepth && !(mainBsdfResult.bRec.sampledType & BSDF::EDelta)) ?
                scene->pdfEmitterDirect(mainDRec) : 0;

        // Power heuristic weights for the following strategies: light sample from base, BSDF sample from base.
        Float mainWeightNumerator = main.pdf * mainBsdfResult.pdf;
        Float mainWeightDenominator = (main.pdf * main.pdf) * ((mainLumPdf * mainLumPdf) + (mainBsdfResult.pdf * mainBsdfResult.pdf));



        if (main.rRec.depth + 1 >= m_config.m_minDepth) {
            Spectrum estimated_radiance = Spectrum(mainBsdfResult.pdf / (D_EPSILON + (mainLumPdf * mainLumPdf) + (mainBsdfResult.pdf * mainBsdfResult.pdf)));
            estimated_radiance *= mainEmitterRadiance * mainBsdfResult.weight * mainBsdfResult.pdf;
            if (estimated_radiance.max() > Float(0) && !branchArguments)
                main.addRadiance(estimated_radiance);
        }

        main.multiply(mainBsdfResult);


        // Construct the offset paths and evaluate emitter hits.

        for (auto& shifted : shiftedRays) {
            Spectrum& main_throughput = shifted.main_throughput;
            Float& main_pdf = shifted.main_pdf;
            Float mainPrevPdf = main_pdf;
            main_pdf *= mainBsdfResult.pdf;
            main_throughput *= mainBsdfResult.weight * mainBsdfResult.pdf;

            Float mainWeightNumerator = mainPrevPdf * mainBsdfResult.pdf;
            Float mainWeightDenominator = (mainPrevPdf * mainPrevPdf) * ((mainLumPdf * mainLumPdf) + (mainBsdfResult.pdf * mainBsdfResult.pdf));

            Spectrum mainContribution(Float(0));
            Spectrum shiftedContribution(Float(0));
            Float weight(0);

            bool postponedShiftEnd = false; // Kills the shift after evaluating the current radiance.


            if (shifted.alive) {
                // The offset path is still good, so it makes sense to continue its construction.
                Float shiftedPreviousPdf = shifted.pdf;

                if (shifted.connection_status == RAY_CONNECTED) {
                    // The offset path keeps following the base path.
                    // As all relevant vertices are shared, we can just reuse the sampled values.
                    Spectrum shiftedBsdfValue = mainBsdfResult.weight * mainBsdfResult.pdf;
                    Float shiftedBsdfPdf = mainBsdfResult.pdf;
                    Float shiftedLumPdf = mainLumPdf;
                    Spectrum shiftedEmitterRadiance = mainEmitterRadiance;

                    // Update throughput and pdf.
                    shifted.throughput *= shiftedBsdfValue;
                    shifted.pdf *= shiftedBsdfPdf;

                    // Power heuristic between light sample from base, BSDF sample from base, light sample from offset, BSDF sample from offset.
                    Float shiftedWeightDenominator = (shiftedPreviousPdf * shiftedPreviousPdf) * ((shiftedLumPdf * shiftedLumPdf) + (shiftedBsdfPdf * shiftedBsdfPdf));
                    weight = mainWeightNumerator / (D_EPSILON + shiftedWeightDenominator + mainWeightDenominator);

                    mainContribution = main_throughput * mainEmitterRadiance;
                    shiftedContribution = shifted.throughput * shiftedEmitterRadiance; // Note: Jacobian baked into .throughput.
                } else if (shifted.connection_status == RAY_RECENTLY_CONNECTED) {
                    // Recently connected - follow the base path but evaluate BSDF to the new direction.
                    Vector3 incomingDirection = normalize(shifted.its.p - main.ray.o);
                    BSDFSamplingRecord bRec(previousMainIts, previousMainIts.toLocal(incomingDirection), previousMainIts.toLocal(main.ray.d), ERadiance);

                    // Note: mainBSDF is the BSDF at previousMainIts, which is the current position of the offset path.

                    EMeasure measure = (mainBsdfResult.bRec.sampledType & BSDF::EDelta) ? EDiscrete : ESolidAngle;

                    Spectrum shiftedBsdfValue = mainBSDF->eval(bRec, measure);

#if defined(FACTOR_MATERIAL)
                    if (main.rRec.depth == 1 && shiftedBsdfValue.max() > Float(0)) // for test
                    {
                        shiftedBsdfValue /= Spectrum(D_EPSILON) + main.pci->factor;
                    }
#endif

                    Float shiftedBsdfPdf = mainBSDF->pdf(bRec, measure);

                    Float shiftedLumPdf = mainLumPdf;
                    Spectrum shiftedEmitterRadiance = mainEmitterRadiance;

                    // Update throughput and pdf.
                    shifted.throughput *= shiftedBsdfValue;
                    shifted.pdf *= shiftedBsdfPdf;

                    shifted.connection_status = RAY_CONNECTED;

                    // Power heuristic between light sample from base, BSDF sample from base, light sample from offset, BSDF sample from offset.
                    Float shiftedWeightDenominator = (shiftedPreviousPdf * shiftedPreviousPdf) * ((shiftedLumPdf * shiftedLumPdf) + (shiftedBsdfPdf * shiftedBsdfPdf));
                    weight = mainWeightNumerator / (D_EPSILON + shiftedWeightDenominator + mainWeightDenominator);

                    mainContribution = main_throughput * mainEmitterRadiance;
                    shiftedContribution = shifted.throughput * shiftedEmitterRadiance; // Note: Jacobian baked into .throughput.
                } else {
                    // Not connected - apply either reconnection or half-vector duplication shift.

                    const BSDF* shiftedBSDF = shifted.its.getBSDF(shifted.ray);
                    // Update the vertex type of the offset path.
                    VertexType shiftedVertexType = getVertexType(shifted, m_config, mainBsdfResult.bRec.sampledType);

                    if (mainVertexType == VERTEX_TYPE_DIFFUSE && mainNextVertexType == VERTEX_TYPE_DIFFUSE && shiftedVertexType == VERTEX_TYPE_DIFFUSE) {
                        // Use reconnection shift.

                        // Optimization: Skip the last raycast and BSDF evaluation for the offset path when it won't contribute and isn't needed anymore.
                        if (!lastSegment || mainHitEmitter || main.rRec.its.hasSubsurface()) {
                            ReconnectionShiftResult shiftResult;
                            bool environmentConnection = false;

                            if (main.rRec.its.isValid()) {
                                // This is an actual reconnection shift.
                                shiftResult = reconnectShift(scene, main.ray.o, main.rRec.its.p, shifted.its.p, main.rRec.its.geoFrame.n, main.ray.time);
                            } else {
                                // This is a reconnection at infinity in environment direction.
                                const Emitter* env = scene->getEnvironmentEmitter();
                                SAssert(env != NULL);

                                environmentConnection = true;
                                shiftResult = environmentShift(scene, main.ray, shifted.its.p);
                            }

                            if (!shiftResult.success) {
                                // Failed to construct the offset path.
                                shifted.alive = false;
#if defined(BACK_PROP_GRAD)
                                if(main.rRec.depth == 1)
                                    shifted.neighbor->merged = false; // failed to merge
#endif
                                goto shift_failed;
                            }

                            Vector3 incomingDirection = -shifted.ray.d;
                            Vector3 outgoingDirection = shiftResult.wo;

                            BSDFSamplingRecord bRec(shifted.its, shifted.its.toLocal(incomingDirection), shifted.its.toLocal(outgoingDirection), ERadiance);

                            // Strict normals check.
                            if (m_config.m_strictNormals && dot(outgoingDirection, shifted.its.geoFrame.n) * Frame::cosTheta(bRec.wo) <= 0) {
                                shifted.alive = false;
                                goto shift_failed;
                            }

                            // Evaluate the BRDF to the new direction.
                            Spectrum shiftedBsdfValue = shiftedBSDF->eval(bRec);
                            Float shiftedBsdfPdf = shiftedBSDF->pdf(bRec);
#if defined(FACTOR_MATERIAL)
                            if (main.rRec.depth == 1) // for test
                            {
                                shiftedBsdfValue /= Spectrum(D_EPSILON) + shiftedBSDF->getDiffuseReflectance(shifted.its);
                            }
#endif
                            // Update throughput and pdf.
                            shifted.throughput *= shiftedBsdfValue * shiftResult.jacobian;
                            shifted.pdf *= shiftedBsdfPdf * shiftResult.jacobian;

                            shifted.connection_status = RAY_RECENTLY_CONNECTED;

                            if (mainHitEmitter || main.rRec.its.hasSubsurface()) {
                                // Also the offset path hit the emitter, as visibility was checked at reconnectShift or environmentShift.

                                // Evaluate radiance to this direction.
                                Spectrum shiftedEmitterRadiance(Float(0));
                                Float shiftedLumPdf = Float(0);

                                if (main.rRec.its.isValid()) {
                                    // Hit an object.
                                    if (mainHitEmitter) {
                                        shiftedEmitterRadiance = main.rRec.its.Le(-outgoingDirection);

                                        // Evaluate the light sampling PDF of the new segment.
                                        DirectSamplingRecord shiftedDRec;
                                        shiftedDRec.p = mainDRec.p;
                                        shiftedDRec.n = mainDRec.n;
                                        shiftedDRec.dist = (mainDRec.p - shifted.its.p).length();
                                        shiftedDRec.d = (mainDRec.p - shifted.its.p) / shiftedDRec.dist;
                                        shiftedDRec.ref = mainDRec.ref;
                                        shiftedDRec.refN = shifted.its.shFrame.n;
                                        shiftedDRec.object = mainDRec.object;

                                        shiftedLumPdf = scene->pdfEmitterDirect(shiftedDRec);
                                    }

                                    // Sub-surface scattering. Note: Should use the same random numbers as the base path!
                                    if (main.rRec.its.hasSubsurface() && (main.rRec.type & RadianceQueryRecord::ESubsurfaceRadiance)) {
                                        shiftedEmitterRadiance += main.rRec.its.LoSub(scene, shifted.rRec.sampler, -outgoingDirection, main.rRec.depth);
                                    }
                                } else {
                                    // Hit the environment.
                                    shiftedEmitterRadiance = mainEmitterRadiance;
                                    shiftedLumPdf = mainLumPdf;
                                }

                                // Power heuristic between light sample from base, BSDF sample from base, light sample from offset, BSDF sample from offset.
                                Float shiftedWeightDenominator = (shiftedPreviousPdf * shiftedPreviousPdf) * ((shiftedLumPdf * shiftedLumPdf) + (shiftedBsdfPdf * shiftedBsdfPdf));
                                weight = mainWeightNumerator / (D_EPSILON + shiftedWeightDenominator + mainWeightDenominator);

                                mainContribution = main_throughput * mainEmitterRadiance;
                                shiftedContribution = shifted.throughput * shiftedEmitterRadiance; // Note: Jacobian baked into .throughput.
                            }
                        }
                    } else {
                        // Use half-vector duplication shift. These paths could not have been sampled by light sampling (by our decision).
                        Vector3 tangentSpaceIncomingDirection = shifted.its.toLocal(-shifted.ray.d);
                        Vector3 tangentSpaceOutgoingDirection;
                        Spectrum shiftedEmitterRadiance(Float(0));

                        const BSDF* shiftedBSDF = shifted.its.getBSDF(shifted.ray);

                        HalfVectorShiftResult shiftResult;
                        EMeasure measure;
                        BSDFSamplingRecord bRec(shifted.its, tangentSpaceIncomingDirection, tangentSpaceOutgoingDirection, ERadiance);

                        Vector3 outgoingDirection;
                        VertexType shiftedVertexType;

                        // Deny shifts between Dirac and non-Dirac BSDFs.
                        bool bothDelta = (mainBsdfResult.bRec.sampledType & BSDF::EDelta) && (shiftedBSDF->getType() & BSDF::EDelta);
                        bool bothSmooth = (mainBsdfResult.bRec.sampledType & BSDF::ESmooth) && (shiftedBSDF->getType() & BSDF::ESmooth);
                        if (!(bothDelta || bothSmooth)) {
                            shifted.alive = false;
                            goto half_vector_shift_failed;
                        }

                        SAssert(fabs(shifted.ray.d.lengthSquared() - 1) < 0.01);

                        // Apply the local shift.
                        shiftResult = halfVectorShift(mainBsdfResult.bRec.wi, mainBsdfResult.bRec.wo, shifted.its.toLocal(-shifted.ray.d), mainBSDF->getEta(), shiftedBSDF->getEta());
                        bRec.wo = shiftResult.wo;

                        if (mainBsdfResult.bRec.sampledType & BSDF::EDelta) {
                            // Dirac delta integral is a point evaluation - no Jacobian determinant!
                            shiftResult.jacobian = Float(1);
                        }

                        if (shiftResult.success) {
                            // Invertible shift, success.
                            shifted.throughput *= shiftResult.jacobian;
                            shifted.pdf *= shiftResult.jacobian;
                            tangentSpaceOutgoingDirection = shiftResult.wo;
                        } else {
                            // The shift is non-invertible so kill it.
                            shifted.alive = false;
                            goto half_vector_shift_failed;
                        }

                        outgoingDirection = shifted.its.toWorld(tangentSpaceOutgoingDirection);

                        // Update throughput and pdf.
                        measure = (mainBsdfResult.bRec.sampledType & BSDF::EDelta) ? EDiscrete : ESolidAngle;


                        shifted.throughput *= shiftedBSDF->eval(bRec, measure);
                        shifted.pdf *= shiftedBSDF->pdf(bRec, measure);

                        if (shifted.pdf == Float(0)) {
                            // Offset path is invalid!
                            shifted.alive = false;
                            goto half_vector_shift_failed;
                        }

                        // Strict normals check to produce the same results as bidirectional methods when normal mapping is used.			
                        if (m_config.m_strictNormals && dot(outgoingDirection, shifted.its.geoFrame.n) * Frame::cosTheta(bRec.wo) <= 0) {
                            shifted.alive = false;
                            goto half_vector_shift_failed;
                        }


                        // Update the vertex type.
                        shiftedVertexType = getVertexType(shifted, m_config, mainBsdfResult.bRec.sampledType);

                        // Trace the next hit point.
                        shifted.ray = Ray(shifted.its.p, outgoingDirection, main.ray.time);

                        if (!scene->rayIntersect(shifted.ray, shifted.its)) {
                            // Hit nothing - Evaluate environment radiance.
                            const Emitter *env = scene->getEnvironmentEmitter();
                            if (!env) {
                                // Since base paths that hit nothing are not shifted, we must be symmetric and kill shifts that hit nothing.
                                shifted.alive = false;
                                goto half_vector_shift_failed;
                            }
                            if (main.rRec.its.isValid()) {
                                // Deny shifts between env and non-env.
                                shifted.alive = false;
                                goto half_vector_shift_failed;
                            }

                            if (mainVertexType == VERTEX_TYPE_DIFFUSE && shiftedVertexType == VERTEX_TYPE_DIFFUSE) {
                                // Environment reconnection shift would have been used for the reverse direction!
                                shifted.alive = false;
                                goto half_vector_shift_failed;
                            }

                            // The offset path is no longer valid after this path segment.
                            shiftedEmitterRadiance = env->evalEnvironment(shifted.ray);
                            postponedShiftEnd = true;
                        } else {
                            // Hit something.

                            if (!main.rRec.its.isValid()) {
                                // Deny shifts between env and non-env.
                                shifted.alive = false;
                                goto half_vector_shift_failed;
                            }

                            VertexType shiftedNextVertexType = getVertexType(shifted, m_config, mainBsdfResult.bRec.sampledType);

                            // Make sure that the reverse shift would use this same strategy!
                            // ==============================================================

                            if (mainVertexType == VERTEX_TYPE_DIFFUSE && shiftedVertexType == VERTEX_TYPE_DIFFUSE && shiftedNextVertexType == VERTEX_TYPE_DIFFUSE) {
                                // Non-invertible shift: the reverse-shift would use another strategy!
                                shifted.alive = false;
                                goto half_vector_shift_failed;
                            }

                            if (shifted.its.isEmitter()) {
                                // Hit emitter.
                                shiftedEmitterRadiance = shifted.its.Le(-shifted.ray.d);
                            }
                            // Sub-surface scattering. Note: Should use the same random numbers as the base path!
                            if (shifted.its.hasSubsurface() && (shifted.rRec.type & RadianceQueryRecord::ESubsurfaceRadiance)) {
                                shiftedEmitterRadiance += shifted.its.LoSub(scene, shifted.rRec.sampler, -shifted.ray.d, main.rRec.depth);
                            }
                        }


half_vector_shift_failed:
                        if (shifted.alive) {
                            // Evaluate radiance difference using power heuristic between BSDF samples from base and offset paths.
                            // Note: No MIS with light sampling since we don't use it for this connection type.
                            weight = main_pdf / (D_EPSILON + shifted.pdf * shifted.pdf + main_pdf * main_pdf);
                            mainContribution = main_throughput * mainEmitterRadiance;
                            shiftedContribution = shifted.throughput * shiftedEmitterRadiance; // Note: Jacobian baked into .throughput.
                        } else {
                            // Handle the failure without taking MIS with light sampling, as we decided not to use it in the half-vector-duplication case.
                            // Could have used it, but so far there has been no need. It doesn't seem to be very useful.
                            weight = Float(1) / (D_EPSILON + main_pdf);
                            mainContribution = main_throughput * mainEmitterRadiance;
                            shiftedContribution = Spectrum(Float(0));

                            // Disable the failure detection below since the failure was already handled.
                            shifted.alive = true;
                            postponedShiftEnd = true;

                            // (TODO: Restructure into smaller functions and get rid of the gotos... Although this may mean having lots of small functions with a large number of parameters.)
                        }
                    }
                }
            }

shift_failed:
            if (!shifted.alive) {
                // The offset path cannot be generated; Set offset PDF and offset throughput to zero.
                weight = mainWeightNumerator / (D_EPSILON + mainWeightDenominator);
                mainContribution = main_throughput * mainEmitterRadiance;
                shiftedContribution = Spectrum((Float) 0);
            }

            // Note: Using also the offset paths for the throughput estimate, like we do here, provides some advantage when a large reconstruction alpha is used,
            // but using only throughputs of the base paths doesn't usually lose by much.

            if (main.rRec.depth + 1 >= m_config.m_minDepth) {
                shifted.addGradient(mainContribution, shiftedContribution, weight, true);
            }

            if (postponedShiftEnd) {
                shifted.alive = false;
            }
        }

        // Stop if the base path hit the environment.
        main.rRec.type = RadianceQueryRecord::ERadianceNoEmission;
        if (!main.rRec.its.isValid() || !(main.rRec.type & RadianceQueryRecord::EIndirectSurfaceRadiance)) {
            break;
        }

        if (main.rRec.depth++ >= m_config.m_rrDepth) {
            /* Russian roulette: try to keep path weights equal to one,
               while accounting for the solid angle compression at refractive
               index boundaries. Stop with at least some probability to avoid
               getting stuck (e.g. due to total internal reflection) */

            Float q = std::min((main.throughput / (D_EPSILON + main.pdf)).max() * main.eta * main.eta, (Float) 0.95f);
            Float rrSample = main.rRec.nextSample1D();
            if (main.rRec.depth - 1 < main.pci->nodes.size() && !branchArguments)
                rrSample = main.pci->nodes[main.rRec.depth - 1].rrSample;
            if (rrSample >= q)
                break;

            for (auto& shifted : shiftedRays) {
                shifted.main_pdf *= q;
                Float w = (shifted.throughput / (D_EPSILON + shifted.pdf) * shifted.getCurrentWeight()).max();
                Float shifted_q = std::min(w, (Float) 0.95f);
                shifted.pdf *= shifted_q;
            }
            main.multiplyPDF(q);
        }
    }

    // Store statistics.
    avgPathLength.incrementBase();
    avgPathLength += main.rRec.depth;
}

/// Returns whether point1 sees point2.

bool testVisibility(const Scene* scene, const Point3& point1, const Point3& point2, Float time) {
    Ray shadowRay;
    shadowRay.setTime(time);
    shadowRay.setOrigin(point1);
    shadowRay.setDirection(point2 - point1);
    shadowRay.mint = Epsilon;
    shadowRay.maxt = (Float) 1.0 - ShadowEpsilon;

    return !scene->rayIntersect(shadowRay);
}

static Float miWeight(Float pdfA, Float pdfB) {
    pdfA *= pdfA;
    pdfB *= pdfB;
    return pdfA / (pdfA + pdfB);
}


/// Returns whether the given ray sees the environment.

bool testEnvironmentVisibility(const Scene* scene, const Ray & ray) {
    const Emitter* env = scene->getEnvironmentEmitter();
    if (!env) {
        return false;
    }

    Ray shadowRay(ray);
    shadowRay.setTime(ray.time);
    shadowRay.setOrigin(ray.o);
    shadowRay.setDirection(ray.d);

    DirectSamplingRecord directSamplingRecord;
    env->fillDirectSamplingRecord(directSamplingRecord, shadowRay);

    shadowRay.mint = Epsilon;
    shadowRay.maxt = ((Float) 1.0 - ShadowEpsilon) * directSamplingRecord.dist;

    return !scene->rayIntersect(shadowRay);
}

void AdaptiveGradientPathIntegrator::MainRayState::spawnShiftedRay(std::vector<ShiftedRayState>& shiftedRays) {
    int activeDpeth = rRec.depth - 1;
    if (activeDpeth >= pci->nodes.size()) return;
    for (auto& neighbor : pci->nodes[activeDpeth].neighbors) {
        neighbor.sampleCount = 1;
        shiftedRays.push_back(ShiftedRayState());
        auto &shifted = shiftedRays.back();
        shifted.ray = neighbor.node->lastRay;
        shifted.rRec = rRec;
        shifted.neighbor = &neighbor;
        shifted.throughput = Spectrum(Float(1));
        shifted.activeDepth = activeDpeth;
        shifted.its = shifted.neighbor->node->its;
    }
}

Spectrum AdaptiveGradientPathIntegrator::Li(const RayDifferential &r, RadianceQueryRecord & rRec) const {
    // Duplicate of MIPathTracer::Li to support sub-surface scattering initialization.

    /* Some aliases and local variables */
    const Scene *scene = rRec.scene;
    Intersection &its = rRec.its;
    RayDifferential ray(r);
    Spectrum Li(0.0f);
    bool scattered = false;

    /* Perform the first ray intersection (or ignore if the
            intersection has already been provided). */
    rRec.rayIntersect(ray);
    ray.mint = Epsilon;

    Spectrum throughput(1.0f);
    Float eta = 1.0f;


    while (rRec.depth <= m_maxDepth || m_maxDepth < 0) {
        if (!its.isValid()) {
            /* If no intersection could be found, potentially return
                    radiance from a environment luminaire if it exists */
            if ((rRec.type & RadianceQueryRecord::EEmittedRadiance)
                    && (!m_hideEmitters || scattered))
                Li += throughput * scene->evalEnvironment(ray);
            break;
        }

        const BSDF *bsdf = its.getBSDF(ray);

        /* Possibly include emitted radiance if requested */
        if (its.isEmitter() && (rRec.type & RadianceQueryRecord::EEmittedRadiance)
                && (!m_hideEmitters || scattered))
            Li += throughput * its.Le(-ray.d);

        /* Include radiance from a subsurface scattering model if requested */
        if (its.hasSubsurface() && (rRec.type & RadianceQueryRecord::ESubsurfaceRadiance))
            Li += throughput * its.LoSub(scene, rRec.sampler, -ray.d, rRec.depth);

        if ((rRec.depth >= m_maxDepth && m_maxDepth > 0)
                || (m_strictNormals && dot(ray.d, its.geoFrame.n)
                * Frame::cosTheta(its.wi) >= 0)) {

            /* Only continue if:
                    1. The current path length is below the specifed maximum
                    2. If 'strictNormals'=true, when the geometric and shading
                        normals classify the incident direction to the same side */
            break;
        }

        /* ==================================================================== */
        /*                     Direct illumination sampling                     */
        /* ==================================================================== */

        /* Estimate the direct illumination if this is requested */
        DirectSamplingRecord dRec(its);

        if (rRec.type & RadianceQueryRecord::EDirectSurfaceRadiance &&
                (bsdf->getType() & BSDF::ESmooth)) {
            Spectrum value = scene->sampleEmitterDirect(dRec, rRec.nextSample2D());
            if (!value.isZero()) {
                const Emitter *emitter = static_cast<const Emitter *> (dRec.object);

                /* Allocate a record for querying the BSDF */
                BSDFSamplingRecord bRec(its, its.toLocal(dRec.d), ERadiance);

                /* Evaluate BSDF * cos(theta) */
                const Spectrum bsdfVal = bsdf->eval(bRec);

                /* Prevent light leaks due to the use of shading normals */
                if (!bsdfVal.isZero() && (!m_strictNormals
                        || dot(its.geoFrame.n, dRec.d) * Frame::cosTheta(bRec.wo) > 0)) {

                    /* Calculate prob. of having generated that direction
                            using BSDF sampling */
                    Float bsdfPdf = (emitter->isOnSurface() && dRec.measure == ESolidAngle)
                            ? bsdf->pdf(bRec) : 0;

                    /* Weight using the power heuristic */
                    Float weight = miWeight(dRec.pdf, bsdfPdf);
                    Li += throughput * value * bsdfVal * weight;
                }
            }
        }

        /* ==================================================================== */
        /*                            BSDF sampling                             */
        /* ==================================================================== */

        /* Sample BSDF * cos(theta) */
        Float bsdfPdf;
        BSDFSamplingRecord bRec(its, rRec.sampler, ERadiance);
        Spectrum bsdfWeight = bsdf->sample(bRec, bsdfPdf, rRec.nextSample2D());
        if (bsdfWeight.isZero())
            break;

        scattered |= bRec.sampledType != BSDF::ENull;

        /* Prevent light leaks due to the use of shading normals */
        const Vector wo = its.toWorld(bRec.wo);
        Float woDotGeoN = dot(its.geoFrame.n, wo);
        if (m_strictNormals && woDotGeoN * Frame::cosTheta(bRec.wo) <= 0)
            break;

        bool hitEmitter = false;
        Spectrum value;

        /* Trace a ray in this direction */
        ray = Ray(its.p, wo, ray.time);
        if (scene->rayIntersect(ray, its)) {
            /* Intersected something - check if it was a luminaire */
            if (its.isEmitter()) {
                value = its.Le(-ray.d);
                dRec.setQuery(ray, its);
                hitEmitter = true;
            }
        } else {
            /* Intersected nothing -- perhaps there is an environment map? */
            const Emitter *env = scene->getEnvironmentEmitter();

            if (env) {
                if (m_hideEmitters && !scattered)
                    break;

                value = env->evalEnvironment(ray);
                if (!env->fillDirectSamplingRecord(dRec, ray))
                    break;
                hitEmitter = true;
            } else {
                break;
            }
        }

        /* Keep track of the throughput and relative
                refractive index along the path */
        throughput *= bsdfWeight;
        eta *= bRec.eta;

        /* If a luminaire was hit, estimate the local illumination and
                weight using the power heuristic */
        if (hitEmitter &&
                (rRec.type & RadianceQueryRecord::EDirectSurfaceRadiance)) {
            /* Compute the prob. of generating that direction using the
                    implemented direct illumination sampling technique */
            const Float lumPdf = (!(bRec.sampledType & BSDF::EDelta)) ?
                    scene->pdfEmitterDirect(dRec) : 0;
            Li += throughput * value * miWeight(bsdfPdf, lumPdf);
        }

        /* ==================================================================== */
        /*                         Indirect illumination                        */
        /* ==================================================================== */

        /* Set the recursive query type. Stop if no surface was hit by the
                BSDF sample or if indirect illumination was not requested */
        if (!its.isValid() || !(rRec.type & RadianceQueryRecord::EIndirectSurfaceRadiance))
            break;
        rRec.type = RadianceQueryRecord::ERadianceNoEmission;

        if (rRec.depth++ >= m_rrDepth) {
            /* Russian roulette: try to keep path weights equal to one,
                    while accounting for the solid angle compression at refractive
                    index boundaries. Stop with at least some probability to avoid
                    getting stuck (e.g. due to total internal reflection) */

            Float q = std::min(throughput.max() * eta * eta, (Float) 0.95f);
            if (rRec.nextSample1D() >= q)
                break;
            throughput /= q;
        }
    }

    return Li;
}

AdaptiveGradientPathIntegrator::ReconnectionShiftResult
AdaptiveGradientPathIntegrator::reconnectShift(const Scene* scene, Point3 mainSourceVertex, Point3 targetVertex, Point3 shiftSourceVertex, Vector3 targetNormal, Float time) {
    ReconnectionShiftResult result;

    // Check visibility of the connection.
    if (!testVisibility(scene, shiftSourceVertex, targetVertex, time)) {
        // Since this is not a light sample, we cannot allow shifts through occlusion.
        result.success = false;
        return result;
    }

    // Calculate the Jacobian.
    Vector3 mainEdge = mainSourceVertex - targetVertex;
    Vector3 shiftedEdge = shiftSourceVertex - targetVertex;

    Float mainEdgeLengthSquared = mainEdge.lengthSquared();
    Float shiftedEdgeLengthSquared = shiftedEdge.lengthSquared();

    Vector3 shiftedWo = -shiftedEdge / sqrt(shiftedEdgeLengthSquared);

    Float mainOpposingCosine = dot(mainEdge, targetNormal) / sqrt(mainEdgeLengthSquared);
    Float shiftedOpposingCosine = dot(shiftedWo, targetNormal);

    Float numerator = std::abs(shiftedOpposingCosine * mainEdgeLengthSquared);
    Float denominator = std::abs(mainOpposingCosine * shiftedEdgeLengthSquared);

    Float jacobian = numerator / (D_EPSILON + denominator);


    // Return the results.
    result.success = true;
    result.jacobian = jacobian;
    result.wo = shiftedWo;

    return result;
}

/// Calculates the outgoing direction of a shift by duplicating the local half-vector.

AdaptiveGradientPathIntegrator::HalfVectorShiftResult
AdaptiveGradientPathIntegrator::halfVectorShift(Vector3 tangentSpaceMainWi, Vector3 tangentSpaceMainWo, Vector3 tangentSpaceShiftedWi, Float mainEta, Float shiftedEta) {
    HalfVectorShiftResult result;

    if (Frame::cosTheta(tangentSpaceMainWi) * Frame::cosTheta(tangentSpaceMainWo) < (Float) 0) {
        // Refraction.

        // Refuse to shift if one of the Etas is exactly 1. This causes degenerate half-vectors.
        if (mainEta == (Float) 1 || shiftedEta == (Float) 1) {
            // This could be trivially handled as a special case if ever needed.
            result.success = false;
            return result;
        }

        // Get the non-normalized half vector.
        Vector3 tangentSpaceHalfVectorNonNormalizedMain;
        if (Frame::cosTheta(tangentSpaceMainWi) < (Float) 0) {
            tangentSpaceHalfVectorNonNormalizedMain = -(tangentSpaceMainWi * mainEta + tangentSpaceMainWo);
        } else {
            tangentSpaceHalfVectorNonNormalizedMain = -(tangentSpaceMainWi + tangentSpaceMainWo * mainEta);
        }

        // Get the normalized half vector.
        Vector3 tangentSpaceHalfVector = normalize(tangentSpaceHalfVectorNonNormalizedMain);

        // Refract to get the outgoing direction.
        Vector3 tangentSpaceShiftedWo = refract(tangentSpaceShiftedWi, tangentSpaceHalfVector, shiftedEta);

        // Refuse to shift between transmission and full internal reflection.
        // This shift would not be invertible: reflections always shift to other reflections.
        if (tangentSpaceShiftedWo.isZero()) {
            result.success = false;
            return result;
        }

        // Calculate the Jacobian.
        Vector3 tangentSpaceHalfVectorNonNormalizedShifted;
        if (Frame::cosTheta(tangentSpaceShiftedWi) < (Float) 0) {
            tangentSpaceHalfVectorNonNormalizedShifted = -(tangentSpaceShiftedWi * shiftedEta + tangentSpaceShiftedWo);
        } else {
            tangentSpaceHalfVectorNonNormalizedShifted = -(tangentSpaceShiftedWi + tangentSpaceShiftedWo * shiftedEta);
        }

        Float hLengthSquared = tangentSpaceHalfVectorNonNormalizedShifted.lengthSquared() / (D_EPSILON + tangentSpaceHalfVectorNonNormalizedMain.lengthSquared());
        Float WoDotH = fabs(dot(tangentSpaceMainWo, tangentSpaceHalfVector)) / (D_EPSILON + fabs(dot(tangentSpaceShiftedWo, tangentSpaceHalfVector)));

        // Output results.
        result.success = true;
        result.wo = tangentSpaceShiftedWo;
        result.jacobian = hLengthSquared * WoDotH;
    } else {
        // Reflection.
        Vector3 tangentSpaceHalfVector = normalize(tangentSpaceMainWi + tangentSpaceMainWo);
        Vector3 tangentSpaceShiftedWo = reflect(tangentSpaceShiftedWi, tangentSpaceHalfVector);

        Float WoDotH = dot(tangentSpaceShiftedWo, tangentSpaceHalfVector) / dot(tangentSpaceMainWo, tangentSpaceHalfVector);
        Float jacobian = fabs(WoDotH);

        result.success = true;
        result.wo = tangentSpaceShiftedWo;
        result.jacobian = jacobian;
    }

    return result;
}

/// Tries to connect the offset path to a the environment emitter.

AdaptiveGradientPathIntegrator::ReconnectionShiftResult
AdaptiveGradientPathIntegrator::environmentShift(const Scene* scene, const Ray& mainRay, Point3 shiftSourceVertex) {
    const Emitter* env = scene->getEnvironmentEmitter();

    ReconnectionShiftResult result;

    // Check visibility of the environment.
    if (!testEnvironmentVisibility(scene, mainRay)) {
        // Sampled by BSDF so cannot accept occlusion.
        result.success = false;
        return result;
    }

    // Return the results.
    result.success = true;
    result.jacobian = Float(1);
    result.wo = mainRay.d;

    return result;
}

AdaptiveGradientPathIntegrator::VertexType
AdaptiveGradientPathIntegrator::getVertexType(const BSDF* bsdf, Intersection& its, const AdaptiveGradientPathTracerConfig& config, unsigned int bsdfType) {
    // Return the lowest roughness value of the components of the vertex's BSDF.
    // If 'bsdfType' does not have a delta component, do not take perfect speculars (zero roughness) into account in this.

    Float lowest_roughness = std::numeric_limits<Float>::infinity();

    bool found_smooth = false;
    bool found_dirac = false;
    for (int i = 0, component_count = bsdf->getComponentCount(); i < component_count; ++i) {
        Float component_roughness = bsdf->getRoughness(its, i);

        if (component_roughness == Float(0)) {
            found_dirac = true;
            if (!(bsdfType & BSDF::EDelta)) {
                // Skip Dirac components if a smooth component is requested.
                continue;
            }
        } else {
            found_smooth = true;
        }

        if (component_roughness < lowest_roughness) {
            lowest_roughness = component_roughness;
        }
    }

    // Roughness has to be zero also if there is a delta component but no smooth components.
    if (!found_smooth && found_dirac && !(bsdfType & BSDF::EDelta)) {
        lowest_roughness = Float(0);
    }

    return getVertexTypeByRoughness(lowest_roughness, config);
}

AdaptiveGradientPathIntegrator::VertexType
AdaptiveGradientPathIntegrator::getVertexType(MainRayState& ray, const AdaptiveGradientPathTracerConfig& config, unsigned int bsdfType) {
    const BSDF* bsdf = ray.rRec.its.getBSDF(ray.ray);
    return getVertexType(bsdf, ray.rRec.its, config, bsdfType);
}

AdaptiveGradientPathIntegrator::VertexType
AdaptiveGradientPathIntegrator::getVertexType(ShiftedRayState& ray, const AdaptiveGradientPathTracerConfig& config, unsigned int bsdfType) {
    const BSDF* bsdf = ray.its.getBSDF(ray.ray);
    return getVertexType(bsdf, ray.rRec.its, config, bsdfType);
}

AdaptiveGradientPathIntegrator::AdaptiveGradientPathIntegrator(const Properties & props)
: MonteCarloIntegrator(props) {
    m_config.m_minDepth = (int) props.getInteger("minDepth", 1);
    m_config.m_shiftThreshold = props.getFloat("shiftThreshold", Float(0.001));
    m_config.m_reconstructAlpha = (Float) props.getFloat("reconstructAlpha", Float(0.2));
    m_config.m_nJacobiIters = (int) props.getInteger("nJacobiIters", 50);
    m_config.m_minMergeDepth = (int) props.getInteger("minMergeDepth", 0);
    m_config.m_maxMergeDepth = (int) props.getInteger("maxMergeDepth", 0);
    m_config.m_usePixelNeighbors = (Float) props.getBoolean("usePixelNeighbors", true);
    m_config.m_batchSize = (int) props.getInteger("batchSize", 1);

    if (m_config.m_reconstructAlpha <= 0.0f)
        Log(EError, "'reconstructAlpha' must be set to a value greater than zero!");
}

AdaptiveGradientPathIntegrator::AdaptiveGradientPathIntegrator(Stream *stream, InstanceManager * manager)
: MonteCarloIntegrator(stream, manager) {
    m_config.m_minDepth = stream->readInt();
    m_config.m_shiftThreshold = stream->readFloat();
    m_config.m_reconstructAlpha = stream->readFloat();
    m_config.m_nJacobiIters = stream->readInt();
    m_config.m_minMergeDepth = stream->readInt();
    m_config.m_maxMergeDepth = stream->readInt();
    m_config.m_usePixelNeighbors = stream->readBool();
    m_config.m_batchSize = stream->readInt();
}

void AdaptiveGradientPathIntegrator::serialize(Stream *stream, InstanceManager * manager) const {
    MonteCarloIntegrator::serialize(stream, manager);
    stream->writeInt(m_config.m_minDepth);
    stream->writeFloat(m_config.m_shiftThreshold);
    stream->writeFloat(m_config.m_reconstructAlpha);
    stream->writeInt(m_config.m_nJacobiIters);
    stream->writeInt(m_config.m_minMergeDepth);
    stream->writeInt(m_config.m_maxMergeDepth);
    stream->writeBool(m_config.m_usePixelNeighbors);
    stream->writeInt(m_config.m_batchSize);
}

std::string AdaptiveGradientPathIntegrator::toString() const {
    std::ostringstream oss;
    oss << "UnstructuredGradientPathTracer[" << endl
            << "  maxDepth = " << m_maxDepth << "," << endl
            << "  rrDepth = " << m_rrDepth << "," << endl
            << "  minDepth = " << m_config.m_minDepth << "," << endl
            << "  shiftThreshold = " << m_config.m_shiftThreshold << endl
            << "  reconstructAlpha = " << m_config.m_reconstructAlpha << endl
            << "  nJacobiIters = " << m_config.m_nJacobiIters << endl
            << "  minMergeDepth = " << m_config.m_minMergeDepth << endl
            << "  maxMergeDepth = " << m_config.m_maxMergeDepth << endl
            << "  usePixelNeighbors = " << m_config.m_usePixelNeighbors << endl
            << "  batchSize = " << m_config.m_batchSize << endl
            << "]";
    return oss.str();
}


MTS_IMPLEMENT_CLASS_S(AdaptiveGradientPathIntegrator, false, MonteCarloIntegrator)
MTS_EXPORT_PLUGIN(AdaptiveGradientPathIntegrator, "Unstructured Gradient Path Integrator");
MTS_NAMESPACE_END
