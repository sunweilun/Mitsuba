#include <mitsuba/bidir/path.h>
#include <mitsuba/bidir/mut_manifold.h>
#include <mitsuba/bidir/manifold.h>

MTS_NAMESPACE_BEGIN

#define MERGE_ONLY

Float Path::miWeightBaseNoSweep_GDVCM(const Scene *scene, const Path &emitterSubpath,
        const PathEdge *connectionEdge, const Path &sensorSubpath,
        const Path&offsetEmitterSubpath, const PathEdge *offsetConnectionEdge,
        const Path &offsetSensorSubpath,
        int s, int t, bool sampleDirect, bool lightImage, Float jDet,
        Float exponent, double geomTermX, double geomTermY, int maxT, float th,
        Float radius, size_t nEmitterPaths, bool merge) {
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

    /* Collect importance transfer area/volume densities from vertices */
    /*The effect of Russian Roulette is ignored here like in the original MIS weight function. Makes stuff easier to implement and has no apparent disadvantage.*/
    pos = 0;
    pdfImp[pos++] = 1.0;

    for (int i = 0; i < s; ++i) {
        pdfImp[pos++] = emitterSubpath.vertex(i)->pdf[EImportance] * emitterSubpath.edge(i)->pdf[EImportance];
    }

    if (merge && !vs->isConnectable()) { // use cached pdf
        pdfImp[pos++] = vs->pdf[EImportance]
                * emitterSubpath.edge(s)->pdf[EImportance];
    } else {
        pdfImp[pos++] = vs->evalPdf(scene, vsPred, vt, EImportance, vsMeasure)
                * connectionEdge->pdf[EImportance];
    }

    if (t > 0) {
        pdfImp[pos++] = vt->evalPdf(scene, vs, vtPred, EImportance, vtMeasure) * sensorSubpath.edge(t - 1)->pdf[EImportance];

        for (int i = t - 1; i > 0; --i) {
            pdfImp[pos++] = sensorSubpath.vertex(i)->pdf[EImportance] * sensorSubpath.edge(i - 1)->pdf[EImportance];
        }
    }

    /* Collect radiance transfer area/volume densities from vertices */
    pos = 0;
    if (s > 0) {
        for (int i = 0; i < s - 1; ++i) {
            pdfRad[pos++] = emitterSubpath.vertex(i + 1)->pdf[ERadiance] * emitterSubpath.edge(i)->pdf[ERadiance];
        }
        if (merge && !vs->isConnectable()) { // use cached pdf
            pdfRad[pos++] = vs->pdf[ERadiance]
                    * emitterSubpath.edge(s - 1)->pdf[ERadiance];
        } else {
            pdfRad[pos++] = vs->evalPdf(scene, vt, vsPred, ERadiance, vsMeasure)
                    * emitterSubpath.edge(s - 1)->pdf[ERadiance];
        }
    }

    pdfRad[pos++] = vt->evalPdf(scene, vtPred, vs, ERadiance, vtMeasure) * connectionEdge->pdf[ERadiance];

    for (int i = t; i > 0; --i) {
        pdfRad[pos++] = sensorSubpath.vertex(i - 1)->pdf[ERadiance] * sensorSubpath.edge(i - 1)->pdf[ERadiance];
    }

    pdfRad[pos++] = 1.0;


    /* When the path contains specular surface interactions, it is possible
    to compute the correct MI weights even without going through all the
    trouble of computing the proper generalized geometric terms (described
    in the SIGGRAPH 2012 specular manifolds paper). The reason is that these
    all cancel out. But to make sure that that's actually true, we need to
    convert some of the area densities in the 'pdfRad' and 'pdfImp' arrays
    into the projected solid angle measure */
    //corresponds to removing geomerty terms!!!
    for (int i = 1; i <= k - 3; ++i) {
        if (!merge && i == s) continue;
        if (!(connectableStrict[i] && !connectableStrict[i + 1]))
            continue;

        const PathVertex *cur = i <= s ? emitterSubpath.vertex(i) : sensorSubpath.vertex(k - i);
        const PathVertex *succ = i + 1 <= s ? emitterSubpath.vertex(i + 1) : sensorSubpath.vertex(k - i - 1);
        const PathEdge *edge = i < s ? emitterSubpath.edge(i) : i == s ? connectionEdge : sensorSubpath.edge(k - i - 1);

        pdfImp[i + 1] *= edge->length * edge->length / std::abs(
                (succ->isOnSurface() ? dot(edge->d, succ->getGeometricNormal()) : 1) *
                (cur->isOnSurface() ? dot(edge->d, cur->getGeometricNormal()) : 1));
    }

    for (int i = k - 1; i >= 3; --i) {
        if (!merge && i - 1 == s) continue;
        if (!(connectableStrict[i] && !connectableStrict[i - 1]))
            continue;

        const PathVertex *cur = i <= s ? emitterSubpath.vertex(i) : sensorSubpath.vertex(k - i);
        const PathVertex *succ = i - 1 <= s ? emitterSubpath.vertex(i - 1) : sensorSubpath.vertex(k - i + 1);
        const PathEdge *edge = i <= s ? emitterSubpath.edge(i - 1) : i - 1 == s ? connectionEdge : sensorSubpath.edge(k - i);

        pdfRad[i - 1] *= edge->length * edge->length / std::abs(
                (succ->isOnSurface() ? dot(edge->d, succ->getGeometricNormal()) : 1) *
                (cur->isOnSurface() ? dot(edge->d, cur->getGeometricNormal()) : 1));
    }

    double initial = 1.0f;

    double sum_p = 0.f, pdf = initial, oPdf = initial;

    /* For VCM: Compute acceptance probability of each vertex. The acceptance probability is 0 if the vertex can not be merged. */
    for (int i = 0; i < n; i++) {
        accProb[i] = Float(0.f);
        bool mergable = connectable[i] && i >= 2 && i <= n - 3;
        if (mergable) {
            accProb[i] = std::min(Float(1.f), pdfImp[i] * mergeArea);
            if (!connectable[i - 1]) accProb[i] = pdfImp[i];
#if defined(MERGE_ONLY)
            accProb[i] = 1e5;
#endif
        }
    }

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
        Float p_conn_merge = ((connectable[p] || isNull[p]) ? 1.0 : 0.0) +
                std::pow(accProb[p+1], exponent) * nEmitterPaths; // for VCM: Now we have 2 ways to sample this path. 1 is for connection, accProb[i] is for merging

        bool allowedToConnect = connectable[p + 1];
        if (allowedToConnect && MIScond_GBDPT(tPrime, p, lightImage))
            sum_p += std::pow(p_i * geomTermX, exponent) * p_conn_merge;

        if (tPrime == t) {
            p_st = std::pow(p_i * geomTermX, exponent) * p_conn_merge;
        }
    }
    double mergeWeight = std::pow(accProb[s + 1], exponent);
    double totalWeight = (connectable[s] ? 1.0 : 0.0) + nEmitterPaths * mergeWeight;
    
    if(sum_p == 0.0) return 0.0;
    
    if (merge) return (Float) mergeWeight > 0.0 ? (mergeWeight * p_st / sum_p / totalWeight) : 0;
    return (Float) (p_st / sum_p) / totalWeight;
}

Float Path::miWeightGradNoSweep_GDVCM(const Scene *scene, const Path &emitterSubpath,
        const PathEdge *connectionEdge, const Path &sensorSubpath,
        const Path&offsetEmitterSubpath, const PathEdge *offsetConnectionEdge,
        const Path &offsetSensorSubpath,
        int s, int t, bool sampleDirect, bool lightImage, Float jDet,
        Float exponent, double geomTermX, double geomTermY, int maxT, float th,
        Float radius, size_t nEmitterPaths, bool merge) {
    int k = s + t + 1, n = k + 1;
    // for vcm
    Float mergeArea = M_PI * radius * radius;
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

    /* Collect importance transfer area/volume densities from vertices */
    /*The effect of Russian Roulette is ignored here like in the original MIS weight function. Makes stuff easier to implement and has no apparent disadvantage.*/
    pos = 0;
    pdfImp[pos] = 1.0;
    offsetPdfImp[pos++] = 1.0;

    for (int i = 0; i < s; ++i) {
        pdfImp[pos] = emitterSubpath.vertex(i)->pdf[EImportance] * emitterSubpath.edge(i)->pdf[EImportance];
        offsetPdfImp[pos++] = offsetEmitterSubpath.vertex(i)->pdf[EImportance] * offsetEmitterSubpath.edge(i)->pdf[EImportance];
    }

    if (merge && !vs->isConnectable()) { // use cached pdf
        pdfImp[pos] = vs->pdf[EImportance]
                * emitterSubpath.edge(s)->pdf[EImportance];
    } else {
        pdfImp[pos] = vs->evalPdf(scene, vsPred, vt, EImportance, vsMeasure)
                * connectionEdge->pdf[EImportance];
    }

    if (merge && !o_vs->isConnectable()) { // use cached pdf
        offsetPdfImp[pos++] = o_vs->pdf[EImportance]
                * offsetEmitterSubpath.edge(s)->pdf[EImportance];
    } else {
        offsetPdfImp[pos++] = o_vs->evalPdf(scene, o_vsPred, o_vt, EImportance, vsMeasure) * offsetConnectionEdge->pdf[EImportance];
    }

    if (t > 0) {
        pdfImp[pos] = vt->evalPdf(scene, vs, vtPred, EImportance, vtMeasure) * sensorSubpath.edge(t - 1)->pdf[EImportance];
        offsetPdfImp[pos++] = o_vt->evalPdf(scene, o_vs, o_vtPred, EImportance, vtMeasure) * offsetSensorSubpath.edge(t - 1)->pdf[EImportance];

        for (int i = t - 1; i > 0; --i) {
            pdfImp[pos] = sensorSubpath.vertex(i)->pdf[EImportance] * sensorSubpath.edge(i - 1)->pdf[EImportance];
            offsetPdfImp[pos++] = offsetSensorSubpath.vertex(i)->pdf[EImportance] * offsetSensorSubpath.edge(i - 1)->pdf[EImportance];
        }
    }

    /* Collect radiance transfer area/volume densities from vertices */
    pos = 0;
    if (s > 0) {
        for (int i = 0; i < s - 1; ++i) {
            pdfRad[pos] = emitterSubpath.vertex(i + 1)->pdf[ERadiance] * emitterSubpath.edge(i)->pdf[ERadiance];
            offsetPdfRad[pos++] = offsetEmitterSubpath.vertex(i + 1)->pdf[ERadiance] * offsetEmitterSubpath.edge(i)->pdf[ERadiance];
        }
        if (merge && !vs->isConnectable()) { // use cached pdf
            pdfRad[pos] = vs->pdf[ERadiance]
                    * emitterSubpath.edge(s - 1)->pdf[ERadiance];
        } else {
            pdfRad[pos] = vs->evalPdf(scene, vt, vsPred, ERadiance, vsMeasure)
                    * emitterSubpath.edge(s - 1)->pdf[ERadiance];
        }
        if (merge && !o_vs->isConnectable()) { // use cached pdf
            offsetPdfRad[pos++] = o_vs->pdf[ERadiance]
                    * offsetEmitterSubpath.edge(s - 1)->pdf[ERadiance];
        } else {
            offsetPdfRad[pos++] = o_vs->evalPdf(scene, o_vt, o_vsPred, ERadiance, vsMeasure)
                    * offsetEmitterSubpath.edge(s - 1)->pdf[ERadiance];
        }

    }

    pdfRad[pos] = vt->evalPdf(scene, vtPred, vs, ERadiance, vtMeasure) * connectionEdge->pdf[ERadiance];
    offsetPdfRad[pos++] = o_vt->evalPdf(scene, o_vtPred, o_vs, ERadiance, vtMeasure) * offsetConnectionEdge->pdf[ERadiance];

    for (int i = t; i > 0; --i) {
        pdfRad[pos] = sensorSubpath.vertex(i - 1)->pdf[ERadiance] * sensorSubpath.edge(i - 1)->pdf[ERadiance];
        offsetPdfRad[pos++] = offsetSensorSubpath.vertex(i - 1)->pdf[ERadiance] * offsetSensorSubpath.edge(i - 1)->pdf[ERadiance];
    }

    pdfRad[pos] = 1.0;
    offsetPdfRad[pos++] = 1.0;

    for (int i = 1; i <= k - 3; ++i) {
        if (!merge && i == s) continue;
        if (!(connectableStrict[i] && !connectableStrict[i + 1]))
            continue;

        const PathVertex *cur = i <= s ? emitterSubpath.vertex(i) : sensorSubpath.vertex(k - i);
        const PathVertex *succ = i + 1 <= s ? emitterSubpath.vertex(i + 1) : sensorSubpath.vertex(k - i - 1);
        const PathEdge *edge = i < s ? emitterSubpath.edge(i) : i == s ? connectionEdge : sensorSubpath.edge(k - i - 1);

        pdfImp[i + 1] *= edge->length * edge->length / std::abs(
                (succ->isOnSurface() ? dot(edge->d, succ->getGeometricNormal()) : 1) *
                (cur->isOnSurface() ? dot(edge->d, cur->getGeometricNormal()) : 1));

        const PathVertex *offset_cur = i <= s ? offsetEmitterSubpath.vertex(i) : offsetSensorSubpath.vertex(k - i);
        const PathVertex *offset_succ = i + 1 <= s ? offsetEmitterSubpath.vertex(i + 1) : offsetSensorSubpath.vertex(k - i - 1);
        const PathEdge *offset_edge = i < s ? offsetEmitterSubpath.edge(i) : offsetSensorSubpath.edge(k - i - 1);

        offsetPdfImp[i + 1] *= offset_edge->length * offset_edge->length / std::abs(
                (offset_succ->isOnSurface() ? dot(offset_edge->d, offset_succ->getGeometricNormal()) : 1) *
                (offset_cur->isOnSurface() ? dot(offset_edge->d, offset_cur->getGeometricNormal()) : 1));
    }

    for (int i = k - 1; i >= 3; --i) {
        if (!merge && i - 1 == s) continue;
        if (!(connectableStrict[i] && !connectableStrict[i - 1]))
            continue;

        const PathVertex *cur = i <= s ? emitterSubpath.vertex(i) : sensorSubpath.vertex(k - i);
        const PathVertex *succ = i - 1 <= s ? emitterSubpath.vertex(i - 1) : sensorSubpath.vertex(k - i + 1);
        const PathEdge *edge = i <= s ? emitterSubpath.edge(i - 1) : i - 1 == s ? connectionEdge : sensorSubpath.edge(k - i);

        pdfRad[i - 1] *= edge->length * edge->length / std::abs(
                (succ->isOnSurface() ? dot(edge->d, succ->getGeometricNormal()) : 1) *
                (cur->isOnSurface() ? dot(edge->d, cur->getGeometricNormal()) : 1));

        const PathVertex *offset_cur = i <= s ? offsetEmitterSubpath.vertex(i) : offsetSensorSubpath.vertex(k - i);
        const PathVertex *offset_succ = i - 1 <= s ? offsetEmitterSubpath.vertex(i - 1) : offsetSensorSubpath.vertex(k - i + 1);
        const PathEdge *offset_edge = i <= s ? offsetEmitterSubpath.edge(i - 1) : offsetSensorSubpath.edge(k - i);

        offsetPdfRad[i - 1] *= offset_edge->length * offset_edge->length / std::abs(
                (offset_succ->isOnSurface() ? dot(offset_edge->d, offset_succ->getGeometricNormal()) : 1) *
                (offset_cur->isOnSurface() ? dot(offset_edge->d, offset_cur->getGeometricNormal()) : 1));
    }

    /* For VCM: Compute acceptance probability of each vertex. The acceptance probability is 0 if the vertex can not be merged. */
    for (int i = 0; i < n; i++) {
        accProb[i] = Float(0.f);
        oAccProb[i] = Float(0.f);
        bool mergable = connectable[i] && i >= 2 && i <= n - 3;
        if (mergable) {
            accProb[i] = std::min(Float(1.f), pdfImp[i] * mergeArea);
            oAccProb[i] = std::min(Float(1.f), offsetPdfImp[i] * mergeArea);
            if (!connectable[i - 1]) accProb[i] = pdfImp[i];
            if (!connectable[i - 1]) oAccProb[i] = offsetPdfImp[i];
#if defined(MERGE_ONLY)
            accProb[i] = oAccProb[i] = 1e3; // for debug
#endif
        }
    }

    double sum_p = 0.f, p_st = 0.f;

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
        
        Float p_conn_merge = ((connectable[p] || isNull[p]) ? 1.0 : 0.0) +
                std::pow(accProb[p+1], exponent) * nEmitterPaths; // for VCM: Now we have 2 ways to sample this path. 1 is for connection, accProb[i] is for merging
        
        Float op_conn_merge = ((connectable[p] || isNull[p]) ? 1.0 : 0.0) +
                std::pow(oAccProb[p+1], exponent) * nEmitterPaths;

        int tPrime = k - p - 1;
        bool allowedToConnect = connectable[p + 1];
        if (allowedToConnect && MIScond_GBDPT(tPrime, p, lightImage))
            sum_p += std::pow(value * geomTermX, exponent) * p_conn_merge + 
                    std::pow(oValue * jDet * geomTermY, exponent) * op_conn_merge;
        if (tPrime == t)
            p_st = std::pow(value * geomTermX, exponent) * p_conn_merge;
    }

    double mergeWeight = std::pow(accProb[s + 1], exponent);
    double totalWeight = (connectable[s] ? 1.0 : 0.0) + nEmitterPaths * mergeWeight;
    
    if(sum_p == 0.0) return 0.0;
    
    if (merge)
        return (Float) (mergeWeight > 0.0 ? (mergeWeight * p_st / sum_p / totalWeight) : 0.0);
    return (Float) (p_st / sum_p / totalWeight);
}

MTS_NAMESPACE_END

