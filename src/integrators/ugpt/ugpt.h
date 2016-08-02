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
#include "ugpt_wr.h"

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
    int m_maxMergeDepth;
};



/* ==================================================================== */
/*                         Integrator                         */

/* ==================================================================== */


struct PrecursorCacheInfo
{
    Point2 samplePos;
    Point2 apertureSample;
    Float timeSample;
    std::vector<Intersection> interList;
    std::vector<Point2> bsdfSampleList;
    PrecursorCacheInfo()
    {
        interList.reserve(3);
        bsdfSampleList.reserve(2);
    }
    void clear()
    {
        interList.clear();
        bsdfSampleList.clear();
    }
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
    void renderBlock(const Scene *scene, const Sensor *sensor, Sampler *sampler, UGPTWorkResult *block,
            const bool &stop, const std::vector< TPoint2<uint8_t> > &points) const;

    void serialize(Stream *stream, InstanceManager *manager) const;
    std::string toString() const;


    /// Used by Mitsuba for initializing sub-surface scattering.
    Spectrum Li(const RayDifferential &ray, RadianceQueryRecord &rRec) const;

    MTS_DECLARE_CLASS()

protected:
    
    std::vector<PrecursorCacheInfo> m_preCacheInfoList;
    std::vector<Spectrum> imageInfo;
    
    void tracePrecursor(const Scene *scene, const Sensor *sensor, Sampler *sampler);
    
    void traceDiff(const Scene *scene, Sensor *sensor, Sampler *sampler);
private:
    UnstructuredGradientPathTracerConfig m_config;

};


MTS_NAMESPACE_END

#endif /* __GBDPT_H */
