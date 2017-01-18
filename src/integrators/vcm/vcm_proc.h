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

MTS_NAMESPACE_BEGIN

/* ==================================================================== */
/*                           Parallel process                           */
/* ==================================================================== */

/**
 * \brief Renders work units (rectangular image regions) using
 * bidirectional path tracing
 */
inline size_t ceil_div(size_t a, size_t b) { return (a+b-1) / b; }

class VCMProcess : public BlockedRenderProcess {
public:
	VCMProcess(const RenderJob *parent, RenderQueue *queue,
		const VCMConfiguration &config);

	inline const VCMWorkResult *getResult() const { return m_result.get(); }

	/// Develop the image
	void develop();
        
        ref<std::vector<CompactBlockPathPool> > getLightPaths() { return &m_result->m_emitterPathPool; }

	/* ParallelProcess impl. */
	void processResult(const WorkResult *wr, bool cancelled);
	ref<WorkProcessor> createWorkProcessor() const;
	void bindResource(const std::string &name, int id);

        void updateConfig(const VCMConfiguration &config) { m_config = config; }
        
	MTS_DECLARE_CLASS()
protected:
	/// Virtual destructor
	virtual ~VCMProcess() { }
private:
	ref<VCMFinalWorkResult> m_result;
	ref<Timer> m_refreshTimer;
	VCMConfiguration m_config;
};

MTS_NAMESPACE_END

#endif /* __VCM_PROC */
