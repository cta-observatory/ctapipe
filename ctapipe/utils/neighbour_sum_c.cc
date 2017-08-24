/*
C extension to sum up the waveforms for a pixel and its neighbours. Used by
ctapipe.image.charge_extractors.NeighbourPeakIntegrator.
*/

#include <iostream>
#include <stdint.h>
#include <stdio.h>

extern "C" void get_sum_array(const float* waveforms, float* sum_array, size_t n_chan, size_t n_pix, size_t n_samples, const uint16_t* nei, size_t nei_length, int lwt)
{
    for (size_t c = 0; c < n_chan; ++c) {
        if (lwt > 0){
            for (size_t p = 0; p < n_pix; ++p) {
                int index = c * n_pix * n_samples + p * n_samples;
                const float* wf = waveforms + index;
                float* sum = sum_array + index;
                for (size_t s = 0; s < n_samples; ++s) {
                    sum[s] += wf[s] * lwt;
                }
            }
        }
        for (size_t ni = 0; ni < nei_length; ++ni) {
            const uint16_t* nei_cur = nei + ni * 2;
            int p = nei_cur[0];
            int n = nei_cur[1];
            int index = c * n_pix * n_samples + p * n_samples;
            int nei_index = c * n_pix * n_samples + n * n_samples;
            const float* wfn = waveforms + nei_index;
            float* sum = sum_array + index;
            for (size_t s = 0; s < n_samples; ++s) {
                sum[s] += wfn[s];
            }
        }
    }
}
