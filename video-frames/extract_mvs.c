/*
 * Copyright (c) 2012 Stefano Sabatini
 * Copyright (c) 2014 Clément Bœsch
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

/**
 * @file libavcodec motion vectors extraction API usage example
 * @example extract_mvs.c
 *
 * Read from input file, decode video stream and print a motion vectors
 * representation to stdout.
 */

/*
 * bracen-video-frames/extract_mvs.c — adapted 2026-09-24 from FFmpeg's doc/examples/extract_mvs.c
 * (MIT licence above). The only change: instead of CSV rows, print ONE JSON line per decoded frame:
 *   {"n":<1-based decode-output index>,"pts":<pts or null>,"type":"I|P|B|?","mvs":[[src_x,src_y,dst_x,dst_y,w,h,source],...]}
 * `source` is -1 when the block is predicted from a PAST frame, +1 from a FUTURE one (AVMotionVector docs).
 * The film's MotionField scene draws exactly these vectors; nothing is drawn that this tool did not print.
 * --summary prints one CSV row per frame instead: n,pts,type,nvec,moving_share,mean_dx,mean_dy,sd_dx (PAST-referencing
 * vectors only, block-area weighted; displacement = dst - src). Used to FIND the pan shot in the film by measurement.
 */
#include <libavutil/motion_vector.h>
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/avutil.h>
#include <math.h>
#include <string.h>

static AVFormatContext *fmt_ctx = NULL;
static AVCodecContext *video_dec_ctx = NULL;
static AVStream *video_stream = NULL;
static const char *src_filename = NULL;
static int video_stream_idx = -1;
static AVFrame *frame = NULL;
static int video_frame_count = 0;
static int summary = 0;  /* --summary: one stats line per frame instead of every vector */

static int decode_packet(const AVPacket *pkt)
{
    int ret = avcodec_send_packet(video_dec_ctx, pkt);
    if (ret < 0) { fprintf(stderr, "send_packet: %s\n", av_err2str(ret)); return ret; }
    while (ret >= 0) {
        ret = avcodec_receive_frame(video_dec_ctx, frame);
        if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) break;
        if (ret < 0) { fprintf(stderr, "receive_frame: %s\n", av_err2str(ret)); return ret; }
        video_frame_count++;
        char t = av_get_picture_type_char(frame->pict_type);
        AVFrameSideData *sd = av_frame_get_side_data(frame, AV_FRAME_DATA_MOTION_VECTORS);
        if (summary) {
            /* per-frame stats over PAST-referencing vectors (source == -1), weighted by block area:
               n, moving share, mean dx/dy of the displacement (dst - src), and the spread of dx. */
            double w = 0, sx = 0, sy = 0, sxx = 0, moving = 0; long n = 0;
            if (sd) {
                const AVMotionVector *mvs = (const AVMotionVector *)sd->data;
                size_t k = sd->size / sizeof(*mvs);
                for (size_t i = 0; i < k; i++) {
                    const AVMotionVector *mv = &mvs[i];
                    if (mv->source != -1) continue;
                    double a = (double)mv->w * mv->h, dx = mv->dst_x - mv->src_x, dy = mv->dst_y - mv->src_y;
                    w += a; sx += a * dx; sy += a * dy; sxx += a * dx * dx; n++;
                    if (dx != 0 || dy != 0) moving += a;
                }
            }
            double mx = w ? sx / w : 0, my = w ? sy / w : 0, vx = w ? sxx / w - mx * mx : 0;
            printf("%d,%lld,%c,%ld,%.4f,%.3f,%.3f,%.3f\n", video_frame_count,
                   frame->pts == AV_NOPTS_VALUE ? -1LL : (long long)frame->pts, t, n,
                   w ? moving / w : 0, mx, my, vx > 0 ? sqrt(vx) : 0);
            av_frame_unref(frame);
            continue;
        }
        if (frame->pts == AV_NOPTS_VALUE) printf("{\"n\":%d,\"pts\":null,\"type\":\"%c\",\"mvs\":[", video_frame_count, t);
        else printf("{\"n\":%d,\"pts\":%lld,\"type\":\"%c\",\"mvs\":[", video_frame_count, (long long)frame->pts, t);
        if (sd) {
            const AVMotionVector *mvs = (const AVMotionVector *)sd->data;
            size_t n = sd->size / sizeof(*mvs);
            for (size_t i = 0; i < n; i++) {
                const AVMotionVector *mv = &mvs[i];
                printf("%s[%d,%d,%d,%d,%d,%d,%d]", i ? "," : "", mv->src_x, mv->src_y, mv->dst_x, mv->dst_y, mv->w, mv->h, mv->source);
            }
        }
        printf("]}\n");
        av_frame_unref(frame);
    }
    return 0;
}

static int open_codec_context(AVFormatContext *fmt, enum AVMediaType type)
{
    const AVCodec *dec = NULL; AVDictionary *opts = NULL;
    int ret = av_find_best_stream(fmt, type, -1, -1, &dec, 0);
    if (ret < 0) { fprintf(stderr, "no %s stream in '%s'\n", av_get_media_type_string(type), src_filename); return ret; }
    int idx = ret; AVStream *st = fmt->streams[idx];
    AVCodecContext *ctx = avcodec_alloc_context3(dec);
    if (!ctx) return AVERROR(EINVAL);
    if ((ret = avcodec_parameters_to_context(ctx, st->codecpar)) < 0) { avcodec_free_context(&ctx); return ret; }
    av_dict_set(&opts, "flags2", "+export_mvs", 0);
    ret = avcodec_open2(ctx, dec, &opts); av_dict_free(&opts);
    if (ret < 0) { fprintf(stderr, "cannot open decoder\n"); return ret; }
    video_stream_idx = idx; video_stream = st; video_dec_ctx = ctx;
    return 0;
}

int main(int argc, char **argv)
{
    int ret = 0; AVPacket *pkt = NULL;
    if (argc == 3 && !strcmp(argv[1], "--summary")) { summary = 1; argv++; argc--; }
    if (argc != 2) { fprintf(stderr, "Usage: %s [--summary] <video>\n", argv[0]); return 1; }
    src_filename = argv[1];
    if (summary) printf("n,pts,type,nvec,moving_share,mean_dx,mean_dy,sd_dx\n");
    if (avformat_open_input(&fmt_ctx, src_filename, NULL, NULL) < 0) { fprintf(stderr, "cannot open %s\n", src_filename); return 1; }
    if (avformat_find_stream_info(fmt_ctx, NULL) < 0) { fprintf(stderr, "no stream info\n"); return 1; }
    if (open_codec_context(fmt_ctx, AVMEDIA_TYPE_VIDEO) < 0 || !video_stream) { ret = 1; goto end; }
    frame = av_frame_alloc(); pkt = av_packet_alloc();
    if (!frame || !pkt) { ret = 1; goto end; }
    while (av_read_frame(fmt_ctx, pkt) >= 0) {
        if (pkt->stream_index == video_stream_idx) ret = decode_packet(pkt);
        av_packet_unref(pkt);
        if (ret < 0) break;
    }
    decode_packet(NULL);
end:
    avcodec_free_context(&video_dec_ctx); avformat_close_input(&fmt_ctx);
    av_frame_free(&frame); av_packet_free(&pkt);
    return ret < 0;
}
