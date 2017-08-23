/*
C extension to sum up the waveforms for a pixel and its neighbours. Used by
ctapipe.image.charge_extractors.NeighbourPeakIntegrator.
*/

#include <iostream>
#include <stdint.h>
#include <stdio.h>

extern "C" void get_sum_array(const float* waveforms, const bool* nei, float* sum_array, size_t n_chan, size_t n_pix, size_t n_samples, int lwt)
{
    for (size_t c = 0; c < n_chan; ++c) {
        for (size_t p = 0; p < n_pix; ++p) {
            int index = c * n_pix * n_samples + p * n_samples;
            const float* wf = waveforms + index;
            float* sum = sum_array + index;
            if (lwt > 0){
                for (size_t s = 0; s < n_samples; ++s) {
                    sum[s] += wf[s] * lwt;
                }
            }
            for (size_t n = 0; n < n_pix; ++n) {
                if (nei[p * n_pix + n]) {
                    int nei_index = c * n_pix * n_samples + n * n_samples;
                    const float* wfn = waveforms + nei_index;
                    for (size_t s = 0; s < n_samples; ++s) {
                        sum[s] += wfn[s];
                    }
                }
            }
        }
    }
}
