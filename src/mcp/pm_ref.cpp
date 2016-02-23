template<int PATCH_W, int USE_PA>
void knn_n(Params *p, BITMAP *a, BITMAP *b,
            VBMP *ann, VBMP *ann_sim, VBMP *annd,
            RegionMasks *amask=NULL, BITMAP *bmask=NULL,
            int level=0, int em_iter=0, RecomposeParams *rp=NULL, int offset_iter=0, int update_type=0, int cache_b=0,
            RegionMasks *region_masks=NULL, int tiles=-1, PRINCIPAL_ANGLE *pa=NULL, int save_first=1) {
  init_xform_tables();
  
  Box box = get_abox(p, a, amask);
  //  poll_server();
  int nn_iter = 0;
  for (; nn_iter < p->nn_iters; nn_iter++) {
      unsigned int iter_seed = rand();

      int ithread = 0;
      int xmin = box.xmin, xmax = box.xmax;
      int ymin = box.ymin + (box.ymax-box.ymin)*ithread/tiles;
      int ymax = box.ymin + (box.ymax-box.ymin)*(ithread+1)/tiles;

      // travel dir
      int ystart = ymin, yfinal = ymax, ychange=1;
      int xstart = xmin, xfinal = xmax, xchange=1;
      if ((nn_iter + offset_iter) % 2 == 1) {
          xstart = xmax-1; xfinal = xmin-1; xchange=-1;
          ystart = ymax-1; yfinal = ymin-1; ychange=-1;
      }
      int dx = -xchange, dy = -ychange;

      int bew = b->w-PATCH_W, beh = b->h-PATCH_W;
      int max_mag = MAX(b->w, b->h);
      int rs_ipart = int(p->rs_iters);
      double rs_fpart = p->rs_iters - rs_ipart;
      int rs_max = p->rs_max;
      if (rs_max > max_mag) { rs_max = max_mag; }

      vector<qtype<int> > v;
      v.reserve(p->knn); // knn
      for (int i = 0; i < p->knn; i++) {
          v.push_back(qtype<int>(0, 0, 0));
      }
#define VARRAY v

      int adata[PATCH_W*PATCH_W];
      for (int y = ystart; y != yfinal; y += ychange) {
          for (int x = xstart; x != xfinal; x += xchange) { // every patch in A

              // Copy patch to adata
              for (int dy0 = 0; dy0 < PATCH_W; dy0++) {
                  int *drow = ((int *) a->line[y+dy0])+x;
                  int *adata_row = adata+(dy0*PATCH_W);
                  for (int dx0 = 0; dx0 < PATCH_W; dx0++) {
                      adata_row[dx0] = drow[dx0];
                  }
              }

              int *p_ann = ann->get(x, y); // NN
              int *p_annd = annd->get(x, y); // dist
              PositionSet pos(p_ann, &v, p->knn); // hash table for positions
              int *p_ann_sim = ann_sim->get(x, y);
              for (int i = 0; i < p->knn; i++) {
                  v[i] = qtype<int>(p_annd[i], p_ann[i], p_ann_sim[i]);
                  pos.insert_nonexistent(INT_TO_X(p_ann[i]), INT_TO_Y(p_ann[i]));
              }

              unsigned int seed = (x | (y<<11)) ^ iter_seed;

              /* Propagate */
              if (p->do_propagate) {
                  /* Propagate x */
                  if ((unsigned) (x+dx) < (unsigned) (ann->w-PATCH_W)) {
                      int *q_ann = ann->get(x+dx, y);
                      int *q_ann_sim = ann_sim->get(x+dx, y);
                      int istart = 0;
                      int n_prop = ((P_BEST_ONLY||P_RAND_ONLY) ? 1: p->knn);
                      for (int i = istart; i < istart+n_prop; i++) {
                          int xpp = INT_TO_X(q_ann[i]), ypp = INT_TO_Y(q_ann[i]);
                          xpp -= dx;
                          XFORM bpos;
                          bpos.x0 = xpp<<16;
                          bpos.y0 = ypp<<16;
                          int spp = 0, tpp = 0;
                          knn_attempt_n<PATCH_W, 0>(v, adata, b, bpos, xpp, ypp, spp, tpp, p, 0, pos);

                      }
                  }

                  /* Propagate y */
                  if ((unsigned) (y+dy) < (unsigned) (ann->h-PATCH_W)) {
                      int *q_ann = ann->get(x, y+dy);
                      int *q_ann_sim = NULL;
                      int istart = 0;

                      int n_prop = ((P_BEST_ONLY||P_RAND_ONLY) ? 1: p->knn);

                      for (int i = istart; i < istart+n_prop; i++) {
                          int xpp = INT_TO_X(q_ann[i]), ypp = INT_TO_Y(q_ann[i]);
                          ypp -= dy;
                          XFORM bpos;
                          bpos.x0 = xpp<<16;
                          bpos.y0 = ypp<<16;
                          int spp = 0, tpp = 0;
                          knn_attempt_n<PATCH_W, 0>(v, adata, b, bpos, xpp, ypp, spp, tpp, p, 0, pos);
                      }
                  }

              }

              /* Random search */
              seed = RANDI(seed);
              int rs_iters = 1-((seed&65535)*(1.0/(65536-1))) < rs_fpart ? rs_ipart + 1: rs_ipart;

              int rs_max_curr = rs_max;


              int h = p->patch_w/2;
              int ymin_clamp = h, xmin_clamp = h;
              int ymax_clamp = BEH+h, xmax_clamp = BEW+h;

              int nchosen = (RS_BEST_ONLY||RS_RAND_ONLY) ? 1: p->knn;

              for (int mag = rs_max_curr; mag >= p->rs_min; mag = int(mag*p->rs_ratio)) {
                  for (int rs_iter = 0; rs_iter < rs_iters; rs_iter++) {
                      int nstart = 0;
                      for (int i = nstart; i < nstart+nchosen; i++) {
                          int xbest = INT_TO_X(VARRAY[i].b), ybest = INT_TO_Y(VARRAY[i].b);
                          int xmin = xbest-mag, xmax = xbest+mag;
                          int ymin = ybest-mag, ymax = ybest+mag;
                          xmax++;
                          ymax++;
                          if (xmin < 0) { xmin = 0; }
                          if (ymin < 0) { ymin = 0; }
                          if (xmax > bew) { xmax = bew; }
                          if (ymax > beh) { ymax = beh; }

                          seed = RANDI(seed);
                          int xpp = xmin+seed%(xmax-xmin);
                          seed = RANDI(seed);
                          int ypp = ymin+seed%(ymax-ymin);
                          XFORM bpos;
                          bpos.x0 = xpp<<16;
                          bpos.y0 = ypp<<16;
                          int spp = 0, tpp = 0;
                          knn_attempt_n<PATCH_W, 0>(v, adata, b, bpos, xpp, ypp, spp, tpp, p, 0, pos);

                      }
                  }
              }
              for (int i = 0; i < p->knn; i++) {
                  p_annd[i] = v[i].a;
                  p_ann[i] = v[i].b;
              }
          } // x
      } // y

      fprintf(stderr, "done with %d iters\n", nn_iter);
  } // nn_iter
  printf("done knn_n, %d iters, rs_max=%d\n", nn_iter, p->rs_max);
}


void knn_attempt_n(vector<qtype<int> > &q, int *adata, BITMAP *b, XFORM bpos, int bx, int by, int bs, int bt, Params *p, int dval_known, PositionSet &pos0) {
  if (q.size() != p->knn) { fprintf(stderr, "q size is wrong (%d, %d)\n", q.size(), p->knn); exit(1); }
  if ((unsigned) (bx) < (unsigned) (b->w-PATCH_W+1) &&
      (unsigned) (by) < (unsigned) (b->h-PATCH_W+1)) {
    int pos = XY_TO_INT(bx, by);
    if (pos0.contains(bx, by, p->knn)) { return; }
    int err = q[0].a;
    int current;
    if (D_KNOWN) {
      current = dval_known;
    } else {
      current = fast_patch_dist<PATCH_W, 0>(adata, b, bpos.x0>>16, bpos.y0>>16, err, p);
    }
    if (current < err) {
        q.push_back(qtype<int>(-1,-1));  // Bug in pop_heap()/push_heap()?  Requires one extra element
        pos0.remove(INT_TO_X(q[0].b), INT_TO_Y(q[0].b));
        pop_heap(&q[0], &q[p->knn]);
        pos0.insert_nonexistent(bx, by);
        q[p->knn-1] = qtype<int>(current, XY_TO_INT(bx, by));
        push_heap(&q[0], &q[p->knn]);
        q.pop_back();
    }
  }
}

template<int PATCH_W>
VBMP *knn_init_dist_n(Params *p, BITMAP *a, BITMAP *b, VBMP *ann, VBMP *ann_sim) {
  init_xform_tables();
  VBMP *ans = new VBMP(a->w, a->h, p->knn);
  clear_to_color(ans, INT_MAX);
  if (ann_sim) { fprintf(stderr, "in mode TRANSLATE_ONLY, expected ann_sim to be NULL\n"); exit(1); }
    #pragma omp parallel for schedule(static,4)
    for (int y = 0; y < AEH; y++) {
      for (int x = 0; x < AEW; x++) {
        int *p_ann = ann->get(x, y);
        int *p_ans = ans->get(x, y);
        int anni = *p_ann;
        int bx = INT_TO_X(anni), by = INT_TO_Y(anni);
        int d = patch_dist_ab<PATCH_W, 0, 0>(p, a, x, y, b, bx, by, INT_MAX, NULL);
        *p_ans = d;
      }
    }
  } else {
    #pragma omp parallel for schedule(static,4)
    for (int y = 0; y < AEH; y++) {
      vector<qtype<int> > v;
      v.reserve(p->knn+1);
      for (int x = 0; x < AEW; x++) {
        int *p_ann = ann->get(x, y);
        int *p_ans = ans->get(x, y);
        v.clear();
        for (int i = 0; i < p->knn; i++) {
          int anni = p_ann[i];
          int bx = INT_TO_X(anni), by = INT_TO_Y(anni);
          int d = patch_dist_ab<PATCH_W, 0, 0>(p, a, x, y, b, bx, by, INT_MAX, NULL);
          v.push_back(qtype<int>(d, anni));
        }
        if (p->knn_algo == KNN_ALGO_HEAP) {
          v.push_back(qtype<int>(0, 0)); // Bug in make_heap()?  Requires one extra element
          make_heap(&v[0], &v[p->knn]);
          v.pop_back();
        }
        if (v.size() != p->knn) { fprintf(stderr, "v size != knn (%d, %d)\n", v.size(), p->knn); exit(1); }
        for (int i = 0; i < p->knn; i++) {
          qtype<int> current = v[i];
          p_ans[i] = current.a;
          p_ann[i] = current.b;
        }
      }
    }
  }
  return ans;
}
