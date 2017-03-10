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

#if !defined(__VCM_H)
#define __VCM_H

#include <mitsuba/mitsuba.h>
#include "vcm_basics.h"
/**
 * When the following is set to "1", the Bidirectional Path Tracer
 * will generate a series of debugging images that split up the final
 * rendering into the weighted contributions of the individual sampling
 * strategies.
 */
//#define VCM_DEBUG 1

MTS_NAMESPACE_BEGIN

/* ==================================================================== */
/*                         Configuration storage                        */
/* ==================================================================== */

/**
 * \brief Stores all configuration parameters of the
 * bidirectional path tracer
 */
struct VCMConfiguration : public VCMConfigBase {
	int blockSize, borderSize;
	bool lightImage;
	bool sampleDirect;
	bool showWeighted;
	size_t sampleCount;
	Vector2i cropSize;
        Float initialRadius;
        Float radiusReductionAlpha;
        bool mergeOnly;
        

	inline VCMConfiguration() { }

	inline VCMConfiguration(Stream *stream) {
		maxDepth = stream->readInt();
		blockSize = stream->readInt();
		lightImage = stream->readBool();
		sampleDirect = stream->readBool();
		showWeighted = stream->readBool();
		sampleCount = stream->readSize();
		cropSize = Vector2i(stream);
		rrDepth = stream->readInt();
                initialRadius = stream->readFloat();
                radiusReductionAlpha = stream->readFloat();
                phExponent = stream->readFloat();
                mergeOnly = stream->readSize();
	}

	inline void serialize(Stream *stream) const {
		stream->writeInt(maxDepth);
		stream->writeInt(blockSize);
		stream->writeBool(lightImage);
		stream->writeBool(sampleDirect);
		stream->writeBool(showWeighted);
		stream->writeSize(sampleCount);
		cropSize.serialize(stream);
		stream->writeInt(rrDepth);
                stream->writeFloat(initialRadius);
                stream->writeFloat(radiusReductionAlpha);
                stream->writeFloat(phExponent);
                stream->writeBool(mergeOnly);
	}

	void dump() const {
		SLog(EDebug, "Bidirectional path tracer configuration:");
		SLog(EDebug, "   Maximum path depth          : %i", maxDepth);
		SLog(EDebug, "   Image size                  : %ix%i",
			cropSize.x, cropSize.y);
		SLog(EDebug, "   Direct sampling strategies  : %s",
			sampleDirect ? "yes" : "no");
		SLog(EDebug, "   Generate light image        : %s",
			lightImage ? "yes" : "no");
		SLog(EDebug, "   Russian roulette depth      : %i", rrDepth);
		SLog(EDebug, "   Block size                  : %i", blockSize);
		SLog(EDebug, "   Number of samples           : " SIZE_T_FMT, sampleCount);
		#if VCM_DEBUG == 1
			SLog(EDebug, "   Show weighted contributions : %s", showWeighted ? "yes" : "no");
		#endif
	}
};

MTS_NAMESPACE_END

#endif /* __VCM_H */
