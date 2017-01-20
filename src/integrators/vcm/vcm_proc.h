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

#if !defined(__VCM_PROC_H)
#define __VCM_PROC_H

#include <mitsuba/render/renderproc.h>
#include <mitsuba/render/renderjob.h>
#include <mitsuba/core/bitmap.h>
#include "vcm_wr.h"


#if defined(MTS_OPENMP)
#define NANOFLANN_USE_OMP
#endif
#include <nanoflann/nanoflann.hpp>

MTS_NAMESPACE_BEGIN

/* ==================================================================== */
/*                           Parallel process                           */
/* ==================================================================== */

/**
 * \brief Renders work units (rectangular image regions) using
 * bidirectional path tracing
 */
inline size_t ceil_div(size_t a, size_t b) {
    return (a + b - 1) / b;
}

class VCMProcess : public BlockedRenderProcess {
    friend class VCMRenderer;
public:
    VCMProcess(const RenderJob *parent, RenderQueue *queue,
            const VCMConfiguration &config);

    inline const VCMWorkResult *getResult() const {
        return m_result.get();
    }

    /// Develop the image
    void develop();

    /* ParallelProcess impl. */
    void processResult(const WorkResult *wr, bool cancelled);
    ref<WorkProcessor> createWorkProcessor() const;
    void bindResource(const std::string &name, int id);

    void updateRadius(int n) 
    {
        m_mergeRadius = 0.2 / pow(n, 1.0 / 3.0);
    }
    
    void clearPhotons() 
    {
        m_photonMap.photons.clear();
    }
    
    typedef nanoflann::KDTreeSingleIndexAdaptor<
    nanoflann::L2_Simple_Adaptor<Float, VCMPhotonMap>,
    VCMPhotonMap, 3 /* dim */ > kd_tree_t;

    void buildPhotonLookupStructure() {
        if(m_photonKDTree) delete m_photonKDTree;
        m_photonKDTree = new kd_tree_t(3, m_photonMap, nanoflann::KDTreeSingleIndexAdaptorParams());
        m_photonKDTree->buildIndex(true);
    }
    
    enum Phase{SAMPLE, EVAL} phase;

    MTS_DECLARE_CLASS()
protected:
    /// Virtual destructor
    virtual ~VCMProcess() {
    }
private:
    ref<VCMWorkResult> m_result;
    ref<Timer> m_refreshTimer;
    VCMConfiguration m_config;

    // for vcm
    size_t nbx, nby; // number of blocks to render
    std::vector<CompactBlockPathPool> m_sensorPathPool;
    std::vector<CompactBlockPathPool> m_emitterPathPool;
    
    VCMPhotonMap m_photonMap;
    kd_tree_t* m_photonKDTree;
    float m_mergeRadius;
    
    std::vector<VCMPhoton> lookupPhotons(const PathVertex* vertex, float radius) {
        std::vector<VCMPhoton> photons;
        if(!vertex->isConnectable()) return photons;
        const Point &position = vertex->getPosition();
        std::vector<std::pair<size_t, Float> > indices_dists;
        nanoflann::RadiusResultSet<Float> resultSet(radius*radius, indices_dists); // we square here because we used squared distance metric
        m_photonKDTree->findNeighbors(resultSet, (Float*) &position, nanoflann::SearchParams());
        for(const std::pair<size_t, Float>& index_dist : indices_dists) {
            photons.push_back(m_photonMap.photons[index_dist.first]);
        }
        return photons;
    }
    
    void extractPath(const VCMPhoton& photon, Path& emitterPath) 
    {
        m_emitterPathPool[photon.blockID].extractPathItem(emitterPath, photon.pointID);
    }
};

MTS_NAMESPACE_END

#endif /* __VCM_PROC */
