#include <mitsuba/bidir/path.h>
#include <mitsuba/bidir/mut_manifold.h>
#include <mitsuba/bidir/manifold.h>

MTS_NAMESPACE_BEGIN

        const Float D_EPSILON = std::numeric_limits<Float>::min(); // to avoid numerical issues

#define USE_GENERALIZED_PDF

void fillPdfList(const Scene* scene, const Path &emitterSubpath, const Path &sensorSubpath, const PathEdge *connectionEdge, int s, int t,
        EMeasure vsMeasure, EMeasure vtMeasure,
        const bool* connectable, bool merge,
        Float* pdfImp, Float* pdfRad, bool* isNull,
        Float* accProb = NULL, const Float& radius = 0.0, int* emitterRefIndirection = NULL, int* sensorRefIndirection = NULL) {
    int k = s + t + 1;
    // emitter: 0, ..., s  sensor: t, ..., 0
    // full:    0, 1, ..., s,s+1, ..., k-1, k
    /* Collect importance transfer area/volume densities from vertices */
    int pos = 0;
    pdfImp[pos++] = 1.0;
    
    auto getEdge = [&](int i) -> const PathEdge* { 
        if(i < s) return emitterSubpath.edge(i); 
        if(i == s) return connectionEdge;
        return sensorSubpath.edge(s+t-i);
    };
    
    auto getVertex = [&](int i) -> const PathVertex* { 
        if(i <= s) return emitterSubpath.vertex(i);
        return sensorSubpath.vertex(k-i);
    };

    const PathVertex
            *vsPred = emitterSubpath.vertexOrNull(s - 1),
            *vtPred = sensorSubpath.vertexOrNull(t - 1),
            *vs = emitterSubpath.vertex(s),
            *vt = sensorSubpath.vertex(t);

    for (int i = 0; i < s; ++i)
        pdfImp[pos++] = emitterSubpath.vertex(i)->pdf[EImportance]
            * emitterSubpath.edge(i)->pdf[EImportance];

    if (merge && !vs->isConnectable()) { // use cached pdf
        pdfImp[pos++] = vs->pdf[EImportance]
                * emitterSubpath.edge(s)->pdf[EImportance];
    } else {
        pdfImp[pos++] = vs->evalPdf(scene, vsPred, vt, EImportance, vsMeasure)
                * connectionEdge->pdf[EImportance];
    }

    if (t > 0) {
        pdfImp[pos++] = vt->evalPdf(scene, vs, vtPred, EImportance, vtMeasure)
                * sensorSubpath.edge(t - 1)->pdf[EImportance];

        for (int i = t - 1; i > 0; --i)
            pdfImp[pos++] = sensorSubpath.vertex(i)->pdf[EImportance]
                * sensorSubpath.edge(i - 1)->pdf[EImportance];
    }

    /* Collect radiance transfer area/volume densities from vertices */
    pos = 0;
    if (s > 0) {
        for (int i = 0; i < s - 1; ++i)
            pdfRad[pos++] = emitterSubpath.vertex(i + 1)->pdf[ERadiance]
                * emitterSubpath.edge(i)->pdf[ERadiance];

        if (merge && !vs->isConnectable()) { // use cached pdf
            pdfRad[pos++] = vs->pdf[ERadiance]
                    * emitterSubpath.edge(s - 1)->pdf[ERadiance];
        } else {
            pdfRad[pos++] = vs->evalPdf(scene, vt, vsPred, ERadiance, vsMeasure)
                    * emitterSubpath.edge(s - 1)->pdf[ERadiance];
        }
    }


    pdfRad[pos++] = vt->evalPdf(scene, vtPred, vs, ERadiance, vtMeasure)
            * connectionEdge->pdf[ERadiance];

    for (int i = t; i > 0; --i)
        pdfRad[pos++] = sensorSubpath.vertex(i - 1)->pdf[ERadiance]
            * sensorSubpath.edge(i - 1)->pdf[ERadiance];

    pdfRad[pos++] = 1.0;


    /* When the path contains specular surface interactions, it is possible
       to compute the correct MI weights even without going through all the
       trouble of computing the proper generalized geometric terms (described
       in the SIGGRAPH 2012 specular manifolds paper). The reason is that these
       all cancel out. But to make sure that that's actually true, we need to
       convert some of the area densities in the 'pdfRad' and 'pdfImp' arrays
       into the projected solid angle measure */
    for (int i = 1; i <= k - 3; ++i) {
        if (!merge && i == s) continue;
        if (!(connectable[i] && !connectable[i + 1]))
            continue;

        const PathVertex *cur = getVertex(i);
        const PathVertex *succ = getVertex(i+1);
        const PathEdge *edge = getEdge(i);

        pdfImp[i + 1] *= edge->length * edge->length / std::abs(
                (succ->isOnSurface() ? dot(edge->d, succ->getGeometricNormal()) : 1) *
                (cur->isOnSurface() ? dot(edge->d, cur->getGeometricNormal()) : 1));
    }

    for (int i = k - 1; i >= 3; --i) {
        if (!merge && i - 1 == s) continue;
        if (!(connectable[i] && !connectable[i - 1]))
            continue;

        const PathVertex *cur = getVertex(i);
        const PathVertex *succ = getVertex(i-1);
        const PathEdge *edge = getEdge(i-1);

        pdfRad[i - 1] *= edge->length * edge->length / std::abs(
                (succ->isOnSurface() ? dot(edge->d, succ->getGeometricNormal()) : 1) *
                (cur->isOnSurface() ? dot(edge->d, cur->getGeometricNormal()) : 1));
    }

    /* One more array sweep before the actual useful work starts -- phew! :)
       "Collapse" edges/vertices that were caused by BSDF::ENull interactions.
       The BDPT implementation is smart enough to connect straight through those,
       so they shouldn't be treated as Dirac delta events in what follows */
    for (int i = 1; i <= k - 3; ++i) {
        if (!connectable[i] || !isNull[i + 1])
            continue;

        int start = i + 1, end = start;
        while (isNull[end + 1])
            ++end;

        if (!connectable[end + 1]) {
            /// The chain contains a non-ENull interaction
            isNull[start] = false;
            continue;
        }

        const PathVertex *before = getVertex(i);
        const PathVertex *after = getVertex(end+1);

        Vector d = before->getPosition() - after->getPosition();
        Float lengthSquared = d.lengthSquared();
        d /= std::sqrt(lengthSquared);

        Float geoTerm = std::abs(
                (before->isOnSurface() ? dot(before->getGeometricNormal(), d) : 1) *
                (after->isOnSurface() ? dot(after->getGeometricNormal(), d) : 1)) / lengthSquared;

        pdfRad[start - 1] *= pdfRad[end] * geoTerm;
        pdfRad[end] = 1;
        pdfImp[start] *= pdfImp[end + 1] * geoTerm;
        pdfImp[end + 1] = 1;

        /* When an ENull chain starts right after the emitter / before the sensor,
           we must keep track of the reference vertex for direct sampling strategies. */
        if (start == 2 && emitterRefIndirection)
            *emitterRefIndirection = end + 1;
        else if (end == k - 2 && sensorRefIndirection)
            *sensorRefIndirection = start - 1;

        i = end;
    }

#if defined(USE_GENERALIZED_PDF)
    // convert pdfs to generalized pdfs, stepping through specular chains. This is needed for correct VCM weights.
    ref<SpecularManifold> sm = new SpecularManifold(scene);

    int chain_start = -1;

    for (int i = 1; i <= k - 2; i++) {
        if (connectable[i] && !connectable[i + 1]) {
            chain_start = i;
            continue;
        } else if (connectable[i] == connectable[i + 1] || chain_start < 0) {
            continue;
        }
        
        Float geoTerm = 1.0;
        // a specular chain (chain_start+1, ..., i, i+1) is found
        if (i <= s) { // chain from emitter path
            geoTerm = sm->G(emitterSubpath, chain_start, i+1);
        } else { // chain from sensor path
            geoTerm = sm->G(sensorSubpath, k-(i+1), k-chain_start);
        }
        pdfImp[i+1] = pdfImp[chain_start+1] * geoTerm;
        pdfImp[chain_start+1] = Float(1.f);
        
        pdfRad[chain_start] = pdfRad[i] * geoTerm;
        pdfRad[i] = Float(1.f);
    }
#endif
    if(!accProb) return;
    // for vcm
    Float mergeArea = M_PI * radius * radius;
    
    /* For VCM: Compute acceptance probability of each vertex. The acceptance probability is 0 if the vertex can not be merged. */
    for (int i = 0; i < k+1; i++) {
        accProb[i] = Float(0.f);
        bool mergable = i >= 2 && i <= k - 2 && connectable[i];
        if (mergable) {
            accProb[i] = std::min(Float(1.f), pdfImp[i] * mergeArea);
#if !defined(USE_GENERALIZED_PDF)
            if (!connectable[i - 1]) accProb[i] = pdfImp[i]; // not a special case anymore in generalized pdf.
#endif
        }
    }
}

Float Path::miWeightVCM(const Scene *scene, const Path &emitterSubpath,
        const PathEdge *connectionEdge, const Path &sensorSubpath,
        int s, int t, bool sampleDirect, bool lightImage, Float exponent,
        Float radius, size_t nEmitterPaths, bool merge) {
    int k = s + t + 1, n = k + 1;

    // if(s != 0 || t != 5) return 0; // for debug
    Float *accProb = (Float *) alloca(n * sizeof (Float));

    const PathVertex
            *vsPred = emitterSubpath.vertexOrNull(s - 1),
            *vtPred = sensorSubpath.vertexOrNull(t - 1),
            *vs = emitterSubpath.vertex(s),
            *vt = sensorSubpath.vertex(t);

    /* pdfImp[i] and pdfRad[i] store the area/volume density of vertex
       'i' when sampled from the adjacent vertex in the emitter
       and sensor direction, respectively. */

    Float ratioEmitterDirect = 0.0f, ratioSensorDirect = 0.0f;
    Float *pdfImp = (Float *) alloca(n * sizeof (Float)),
            *pdfRad = (Float *) alloca(n * sizeof (Float));
    bool *connectable = (bool *) alloca(n * sizeof (bool)),
            *isNull = (bool *) alloca(n * sizeof (bool));


    /* Keep track of which vertices are connectable / null interactions */
    int pos = 0;
    for (int i = 0; i <= s; ++i) {
        const PathVertex *v = emitterSubpath.vertex(i);
        connectable[pos] = v->isConnectable();
        isNull[pos] = v->isNullInteraction() && !connectable[pos];
        pos++;
    }

    for (int i = t; i >= 0; --i) {
        const PathVertex *v = sensorSubpath.vertex(i);
        connectable[pos] = v->isConnectable();
        isNull[pos] = v->isNullInteraction() && !connectable[pos];
        pos++;
    }

    if (k <= 3)
        sampleDirect = false;

    EMeasure vsMeasure = EArea, vtMeasure = EArea;
    if (sampleDirect) {
        /* When direct sampling is enabled, we may be able to create certain
           connections that otherwise would have failed (e.g. to an
           orthographic camera or a directional light source) */
        const AbstractEmitter *emitter = (s > 0 ? emitterSubpath.vertex(1) : vt)->getAbstractEmitter();
        const AbstractEmitter *sensor = (t > 0 ? sensorSubpath.vertex(1) : vs)->getAbstractEmitter();

        EMeasure emitterDirectMeasure = emitter->getDirectMeasure();
        EMeasure sensorDirectMeasure = sensor->getDirectMeasure();

        connectable[0] = emitterDirectMeasure != EDiscrete && emitterDirectMeasure != EInvalidMeasure;
        connectable[1] = emitterDirectMeasure != EInvalidMeasure;
        connectable[k - 1] = sensorDirectMeasure != EInvalidMeasure;
        connectable[k] = sensorDirectMeasure != EDiscrete && sensorDirectMeasure != EInvalidMeasure;

        /* The following is needed to handle orthographic cameras &
           directional light sources together with direct sampling */
        if (t == 1)
            vtMeasure = sensor->needsDirectionSample() ? EArea : EDiscrete;
        else if (s == 1)
            vsMeasure = emitter->needsDirectionSample() ? EArea : EDiscrete;
    }

    int emitterRefIndirection = 2, sensorRefIndirection = k - 2;


    fillPdfList(scene, emitterSubpath, sensorSubpath, connectionEdge, s, t, vsMeasure, vtMeasure,
            connectable, merge, pdfImp, pdfRad, isNull, accProb, radius, 
            &emitterRefIndirection, &sensorRefIndirection);

    double initial = 1.0f;

    /* When direct sampling strategies are enabled, we must
       account for them here as well */
    if (sampleDirect) {
        /* Direct connection probability of the emitter */
        const PathVertex *sample = s > 0 ? emitterSubpath.vertex(1) : vt;
        const PathVertex *ref = emitterRefIndirection <= s
                ? emitterSubpath.vertex(emitterRefIndirection) : sensorSubpath.vertex(k - emitterRefIndirection);
        EMeasure measure = sample->getAbstractEmitter()->getDirectMeasure();

        if (connectable[1] && connectable[emitterRefIndirection])
            ratioEmitterDirect = ref->evalPdfDirect(scene, sample, EImportance,
                measure == ESolidAngle ? EArea : measure) / pdfImp[1];

        /* Direct connection probability of the sensor */
        sample = t > 0 ? sensorSubpath.vertex(1) : vs;
        ref = sensorRefIndirection <= s ? emitterSubpath.vertex(sensorRefIndirection)
                : sensorSubpath.vertex(k - sensorRefIndirection);
        measure = sample->getAbstractEmitter()->getDirectMeasure();

        if (connectable[k - 1] && connectable[sensorRefIndirection])
            ratioSensorDirect = ref->evalPdfDirect(scene, sample, ERadiance,
                measure == ESolidAngle ? EArea : measure) / pdfRad[k - 1];

        if (s == 1)
            initial /= ratioEmitterDirect;
        else if (t == 1)
            initial /= ratioSensorDirect;
    }

    double weight = 1, pdf = initial;

    /* With all of the above information, the MI weight can now be computed.
       Since the goal is to evaluate the power heuristic, the absolute area
       product density of each strategy is interestingly not required. Instead,
       an incremental scheme can be used that only finds the densities relative
       to the (s,t) strategy, which can be done using a linear sweep. For
       details, refer to the Veach thesis, p.306. */
    
    auto num_conn_shemes = [&connectable, &nEmitterPaths, &k](int i) {
        return Float(connectable[i] ? 1 : 0);
    };

    double base_prob_exp = num_conn_shemes(s) + pow(accProb[s + 1], exponent) * nEmitterPaths;

    for (int i = s + 1; i < k; ++i) {
        double prob_exp = num_conn_shemes(i) + pow(accProb[i + 1], exponent) * nEmitterPaths;
        double next = pdf * (double) pdfImp[i] / (double) pdfRad[i],
                value = next;

        if (sampleDirect) {
            if (i == 1)
                value *= ratioEmitterDirect;
            else if (i == sensorRefIndirection)
                value *= ratioSensorDirect;
        }


        int tPrime = k - i - 1;
        if ((connectable[i + 1] || isNull[i + 1]) && (lightImage || tPrime > 1)) {
            weight += pow(value, exponent) * prob_exp / base_prob_exp;
        }

        pdf = next;
    }

    /* As above, but now compute pdf[i] with i<s (this is done by
       evaluating the inverse of the previous expressions). */
    pdf = initial;
    for (int i = s - 1; i >= 0; --i) {
        double prob_exp = num_conn_shemes(i) + pow(accProb[i + 1], exponent) * nEmitterPaths;
        double next = pdf * (double) pdfRad[i + 1] / (double) pdfImp[i + 1],
                value = next;

        if (sampleDirect) {
            if (i == 1)
                value *= ratioEmitterDirect;
            else if (i == sensorRefIndirection)
                value *= ratioSensorDirect;
        }

        int tPrime = k - i - 1;
        if ((connectable[i + 1] || isNull[i + 1]) && (lightImage || tPrime > 1)) {
            weight += pow(value, exponent) * prob_exp / base_prob_exp;
        }

        pdf = next;
    }

    Float total_weight = 1.0 / weight;
    Float w_merge = pow(accProb[s + 1], exponent) / base_prob_exp;
    
    if (merge) return total_weight * w_merge;
    return total_weight * num_conn_shemes(s) / base_prob_exp;
}

Float Path::miWeightBaseNoSweep_GDVCM(const Scene *scene, const Path &emitterSubpath,
        const PathEdge *connectionEdge, const Path &sensorSubpath,
        const Path&offsetEmitterSubpath, const PathEdge *offsetConnectionEdge,
        const Path &offsetSensorSubpath,
        int s, int t, bool sampleDirect, bool lightImage, Float jDet,
        Float exponent, double geomTermX, double geomTermY, int maxT, float th,
        Float radius, size_t nEmitterPaths, bool merge, bool merge_only) {
    int k = s + t + 1, n = k + 1;
    // for vcm
    Float mergeArea = M_PI * radius * radius;
    Float *accProb = (Float *) alloca(n * sizeof (Float));

    const PathVertex
            *vsPred = emitterSubpath.vertexOrNull(s - 1),
            *vtPred = sensorSubpath.vertexOrNull(t - 1),
            *vs = emitterSubpath.vertex(s),
            *vt = sensorSubpath.vertex(t);

    /* pdfImp[i] and pdfRad[i] store the area/volume density of vertex
    'i' when sampled from the adjacent vertex in the emitter
    and sensor direction, respectively. */

    Float ratioEmitterDirect = 0.0f;
    Float ratioSensorDirect = 0.0f;
    Float *pdfImp = (Float *) alloca(n * sizeof (Float));
    Float *pdfRad = (Float *) alloca(n * sizeof (Float));
    bool *connectable = (bool *)alloca(n * sizeof (bool));
    bool *connectableStrict = (bool *)alloca(n * sizeof (bool));
    bool *isNull = (bool *)alloca(n * sizeof (bool));
    bool *isEmitter = (bool *)alloca(n * sizeof (bool));

    /* Keep track of which vertices are connectable / null interactions */
    // a perfectly specular interaction is *not* connectable!
    int pos = 0;
    for (int i = 0; i <= s; ++i) {
        const PathVertex *v = emitterSubpath.vertex(i);
        connectable[pos] = Path::isConnectable_GBDPT(v, th);
        connectableStrict[pos] = v->isConnectable();
        isNull[pos] = v->isNullInteraction() && !connectable[pos];
        isEmitter[pos] = v->isEmitterSample();
        pos++;
    }

    for (int i = t; i >= 0; --i) {
        const PathVertex *v = sensorSubpath.vertex(i);
        connectable[pos] = Path::isConnectable_GBDPT(v, th);
        connectableStrict[pos] = v->isConnectable();
        isNull[pos] = v->isNullInteraction() && !connectable[pos];
        isEmitter[pos] = v->isEmitterSample();
        pos++;
    }

    EMeasure vsMeasure = EArea, vtMeasure = EArea;

    fillPdfList(scene, emitterSubpath, sensorSubpath, connectionEdge, s, t, vsMeasure, vtMeasure,
            connectable, merge, pdfImp, pdfRad, isNull, accProb, radius);

    double sum_p = 0.f;
    
    auto num_conn_shemes = [&connectable, &isNull, &nEmitterPaths, &k, &merge_only](int i) {
        if((connectable[i] || isNull[i]) && !merge_only) {
            return Float(1);
        }
        return Float(0);
    };

    /* No linear sweep */
    double p_i, p_st;
    for (int p = 0; p < s + t + 1; ++p) {
        p_i = 1.f;

        for (int i = 1; i < p + 1; ++i) {
            p_i *= pdfImp[i];
        }

        for (int i = p + 1; i < s + t + 1; ++i) {
            p_i *= pdfRad[i];
        }

        int tPrime = k - p - 1;
        Float p_conn_merge = num_conn_shemes(p) +
                std::pow(accProb[p + 1], exponent) * nEmitterPaths; // for VCM: Now we have 2 ways to sample this path. 1 is for connection, accProb[i] is for merging

        bool allowedToConnect = connectable[p + 1];
        if (allowedToConnect && MIScond_GBDPT(tPrime, p, lightImage))
            sum_p += std::pow(p_i * geomTermX, exponent) * p_conn_merge;

        if (tPrime == t) {
            p_st = std::pow(p_i * geomTermX, exponent) * p_conn_merge;
        }
    }
    double mergeWeight = std::pow(accProb[s + 1], exponent);
    double totalWeight = num_conn_shemes(s) + nEmitterPaths * mergeWeight + D_EPSILON;

    if (sum_p == 0.0) return 0.0;

    if (merge) return (Float) mergeWeight > 0.0 ? (mergeWeight * p_st / sum_p / totalWeight) : 0;
    return (Float) (p_st / sum_p) / totalWeight * num_conn_shemes(s);
}

Float Path::miWeightGradNoSweep_GDVCM(const Scene *scene, const Path &emitterSubpath,
        const PathEdge *connectionEdge, const Path &sensorSubpath,
        const Path&offsetEmitterSubpath, const PathEdge *offsetConnectionEdge,
        const Path &offsetSensorSubpath,
        int s, int t, bool sampleDirect, bool lightImage, Float jDet,
        Float exponent, double geomTermX, double geomTermY, int maxT, float th,
        Float radius, size_t nEmitterPaths, bool merge, bool merge_only) {
    int k = s + t + 1, n = k + 1;
    // for vcm
    Float *accProb = (Float *) alloca(n * sizeof (Float));
    Float *oAccProb = (Float *) alloca(n * sizeof (Float));

    const PathVertex
            *vsPred = emitterSubpath.vertexOrNull(s - 1),
            *vtPred = sensorSubpath.vertexOrNull(t - 1),
            *vs = emitterSubpath.vertex(s),
            *vt = sensorSubpath.vertex(t);

    const PathVertex
            *o_vsPred = offsetEmitterSubpath.vertexOrNull(s - 1),
            *o_vtPred = offsetSensorSubpath.vertexOrNull(t - 1),
            *o_vs = offsetEmitterSubpath.vertex(s),
            *o_vt = offsetSensorSubpath.vertex(t);

    /* pdfImp[i] and pdfRad[i] store the area/volume density of vertex
    'i' when sampled from the adjacent vertex in the emitter
    and sensor direction, respectively. */

    Float ratioEmitterDirect = 0.0f;
    Float ratioSensorDirect = 0.0f;
    Float *pdfImp = (Float *) alloca(n * sizeof (Float));
    Float *pdfRad = (Float *) alloca(n * sizeof (Float));
    Float ratioOffsetEmitterDirect = 0.0f;
    Float ratioOffsetSensorDirect = 0.0f;
    Float *offsetPdfImp = (Float *) alloca(n * sizeof (Float));
    Float *offsetPdfRad = (Float *) alloca(n * sizeof (Float));
    bool *connectable = (bool *)alloca(n * sizeof (bool));
    bool *connectableStrict = (bool *)alloca(n * sizeof (bool));
    bool *isNull = (bool *)alloca(n * sizeof (bool));
    bool *isEmitter = (bool *)alloca(n * sizeof (bool));

    /* Keep track of which vertices are connectable / null interactions */
    // a perfectly specular interaction is *not* connectable!
    int pos = 0;
    for (int i = 0; i <= s; ++i) {
        const PathVertex *v = emitterSubpath.vertex(i);
        connectable[pos] = Path::isConnectable_GBDPT(v, th);
        connectableStrict[pos] = v->isConnectable();
        isNull[pos] = v->isNullInteraction() && !connectable[pos];
        isEmitter[pos] = v->isEmitterSample();
        pos++;
    }

    for (int i = t; i >= 0; --i) {
        const PathVertex *v = sensorSubpath.vertex(i);
        connectable[pos] = Path::isConnectable_GBDPT(v, th);
        connectableStrict[pos] = v->isConnectable();
        isNull[pos] = v->isNullInteraction() && !connectable[pos];
        isEmitter[pos] = v->isEmitterSample();
        pos++;
    }

    EMeasure vsMeasure = EArea, vtMeasure = EArea;

    fillPdfList(scene, emitterSubpath, sensorSubpath, connectionEdge, s, t, vsMeasure, vtMeasure,
            connectable, merge, pdfImp, pdfRad, isNull, accProb, radius);

    fillPdfList(scene, offsetEmitterSubpath, offsetSensorSubpath, offsetConnectionEdge, s, t, vsMeasure, vtMeasure,
            connectable, merge, offsetPdfImp, offsetPdfRad, isNull, oAccProb, radius);

    double sum_p = 0.f, p_st = 0.f;
    
    auto num_conn_shemes = [&connectable, &isNull, &nEmitterPaths, &k, &merge_only](int i) {
        if((connectable[i] || isNull[i]) && !merge_only) {
            return Float(1);
        }
        return Float(0);
    };

    /* No linear sweep */
    double value, oValue;
    for (int p = 0; p < s + t + 1; ++p) {
        value = 1.f;
        oValue = 1.f;

        for (int i = 1; i < p + 1; ++i) {
            value *= pdfImp[i];
            oValue *= offsetPdfImp[i];
        }

        for (int i = p + 1; i < s + t + 1; ++i) {
            value *= pdfRad[i];
            oValue *= offsetPdfRad[i];
        }

        Float p_conn_merge = num_conn_shemes(p) +
                std::pow(accProb[p + 1], exponent) * nEmitterPaths; // for VCM: Now we have 2 ways to sample this path. 1 is for connection, accProb[i] is for merging

        Float op_conn_merge = num_conn_shemes(p) +
                std::pow(oAccProb[p + 1], exponent) * nEmitterPaths;

        int tPrime = k - p - 1;
        bool allowedToConnect = connectable[p + 1];
        if (allowedToConnect && MIScond_GBDPT(tPrime, p, lightImage))
            sum_p += std::pow(value * geomTermX, exponent) * p_conn_merge +
                std::pow(oValue * jDet * geomTermY, exponent) * op_conn_merge;
        if (tPrime == t)
            p_st = std::pow(value * geomTermX, exponent) * p_conn_merge;
    }

    double mergeWeight = std::pow(accProb[s + 1], exponent);
    double totalWeight = num_conn_shemes(s) + nEmitterPaths * mergeWeight + D_EPSILON;

    if (sum_p == 0.0) return 0.0;

    if (merge)
        return (Float) (mergeWeight > 0.0 ? (mergeWeight * p_st / sum_p / totalWeight) : 0.0);
    return (Float) (p_st * num_conn_shemes(s) / sum_p / totalWeight);
}

MTS_NAMESPACE_END

