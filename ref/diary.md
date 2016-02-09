Motion Comparison (C++) - 2014
==============================
Michael Gharbi <gharbi@mit.edu>

2014

2014-01-03
----------

Representation is non unique
motion compensated median filtering ?
motion compensated temporal smoothness? ghosts and doubles


2014-01-07
----------

* Forward warping
* Occlusion/Disocclusion -> divergence of the flow
* Forward in gradient domain, then Poisson reconstruction  for better results ?
* Timewarp -> speedup/slowdown ? simple multiplication of the offset is wrong,
time reversal etc


Examples
* Sports video Guha
* Sewing machine
* Impact vs strength Slowmo: golf
* Copper tube, sawed a bit
* Electric saw, nail gun, drill: on different material
* Sewing machine
* Loaded vs empty, different kind of resistance
* Piano hammer, drum
* Robotic arm
* Friction w/wo lubrication
* Lego cars

MIT hobby shop
CSAIL shop

Warp code first
Warp field next

2014-01-08
----------

Time interpolation of the frames? Use optical flow?.

Applications:
- Temporally align, to visually assess spatial differences
- Temporally align + amplify spatial differences
- Amplify spatial differences
- Amplify time offset
- Time stretch
- Amplify velocity difference

What kind of filtering/Processing?


2014-01-10
----------

* Warping code works reasonably
* HobbyShop on Tuesday

Concerns:
* is the estimation correct? are we making stuff up?
* resolution vs. computation time: but resolution is needed when dealing with
small motions.

Cool examples, what kind of validation for the paper? Synthetic?
What would sell the point?

Paper:
* Write optical flow related works. 
* Write rendering and reconstruction section
* Write applications section

Advanced median filter causes problem with occlusion on the forward warping.

* continuous light + stand

2014-01-12
----------

Filter out areas where the flow doesnt change over time -> get rid of DC
component, mean
Estimate variance -> noise classification: amplify only over a and below some
thresholds ? time analysis => optical flow 

2014-01-13 - Meeting with Ken Stone (Hobby Shop)
----------

* Ask James W. Bales from the MIT Edgerton Center:

Dear Mr. Bales,

     I am a second PhD student at MIT CSAIl working with Fredo Durand. I am
     currently working on a project that seeks to compare motions using video
     data. We try to amplify the small differences that occur between similar
     motions.

     We have a paper submission deadline on Jan. 21st and I am currently looking
     for data on which to run my algorithm. I met with Ken Stone from the Hobby
     Shop this morning and he suggested that you might have some useful videos
     given your expertise in high-speed imaging. 

     If you'd be willing to share some of them, that would be really helpul for
     my research. I am more specifically looking for interesting motions of rigid bodies,
     that are similar yet possess subtle differences not necessarily obvious to
     the eye, and shot under similar lighting conditions. An example would be a
     milling machine working on different types of materials.

     I am looking forward to read you.

     Best regards,

     Michael Gharbi
     gharbi@mit.edu
     --------------
      857 210 9559


2014-01-16 - Meeting with Fredo
----------

* MRI data
* MitMuseum
* Slowmo bike
* Newton balls
* Tomorrow construction site?
* HobbyShop not very useful, light variations, mostly rotating stuffs
* Sewing machine not yet processed

* Reconstruction is messy. Time filtering? DC component? Tried naive filtering,
    without success
* Use optical flow for long range time series to be able to filter?
* Time processing has the same problem: filtering non corresponding pixels

* Going slow on the writeup, making figures.

* No Poission Reconstruction yet

Paper: 
* nPages?
* which figures?
* experimental validation: comparison to motion magnification's output?

Viz:
* edges viz

Align MRI

2014-01-24
----------

Color weighted regularization

2014-02-06
----------

* Flow weighted regularization
* Increase regularization, background influence is high because match is perfect,
force the flow to be close to zero everywhere.

Paulina, Rick Szeliski opt flow occlusion

2014-02-18
----------

* Viz: line overlay, vertex/edge representation w.links, forward warp vertices,
    keep edges
* Params dump
* MedFilt
* Poisson reconstruction
* Try uvw on optical flow: pb not enough texture, as an added channel?
* MATLAB .stw reader

2014-02-20
----------

* lambdaT causes most of the ghosting and problems

2014-02-28 - Meeting with Fredo
----------

* Tried optical flow as data term
* Optical flow only: not enough texture
* Optical flow as another channel: no much difference
* Optical flow as weighting term: unsucessful
* Motion aware frame interpolation? Might become costly very fast in we have a
spatiotemporally varying space warping map that doesn't map discrete offsets.

* PB1: Temporal interpolation causes ghosting and is a major problem in most cases.
* PB2: We need temporal registration to do any meaningful processing
* pb3: accuracy of the registration is so-so

* Bill's suggestion on motion mag with large displacement
* Compare the phase difference btw two videos
* In any case: long-range temporal registration, tracking vs. opt flow:
difficult. Trying particle video, semi long-range, semi-dense, temporal
processing on the particles then interpolation: we can use the phase, the
warping field... But accuracy is doubtful

* do we need, motion aware temporal regularization in the general case
(mvt> 1px/frame) ?

* Plan: make particle video work, temporally process particles (u,v,w), or phase
, then interpolate.


2014-03-05
----------

* Occlusion cost via L2 color difference
* Layers

* Color weighted gradient
* Edge aware regul/mask/layers
* IIR fiters
* Examples: Mindstorm robot legos

* Motion compensated filtering?

2014-03-06
----------

* Occlusion: derailleur example, where not even try to match it
* Large displacement: out of the linearization bound
* Bilateral or weighted median filtering for edges preservation: Color, flow,
occlusion, distance
* Coarse to fine get stuck in local optimum?
* Amplitude of motion too high for standard optical flow?
* Augment with SIFT/patchmatch matching?

* Motion compensated regularization: if object moves fast against background,
displacement gets regularized to zeros.

2014-03-14
----------

* Data !
* Mostly reading, QPBO, GraphCut/MRF for labeling
* Started implementing patch matching (PatchMatch vs CSH)
    - preliminary shows 2d patches are not discriminative enough
    - What kind of transform for the labeling/clustering problem?
    - affine in time, similarity in space?
    - global 3d ?
    - extent with sift/hog-like features?
* Action recognition litterature

2014-03-21
----------

* For now extracting only dominant translations from the NNF
* Super-flow
* Bug in graphcut: output is a uniform field..
* Dominant translations close to 0:
    - treat space and time separately?
    - more candidates ? (40, expensive)
    - process each frame independently?
* Looks like the time offset is unusable., not informative biases the dominant translation parts
* Bug? time offset seems to be monotically decreasing? mapping to same frame ?
* Bug in Graphcut segmentation

Legos  and robots
ST-slices derailleur golfball what disappear
order of magnitude short time events, set pyramid time scale accordlingly

Using different metrics for correspondence business 

Using time gradient in dataterm
space time slices debug -> is motion visible at small scales 

* Examples: swing, pool


2014-03-26
----------

* Adding temporal derivatives in the data term, seems to help
* Decouple u,v,w in dims x,y,t for the regularization, seems to help, axis
aligned continuity boundaries

* Using flow direction for regularization: seems to harm. Non symmetric MRF, etc
nodes not connected?

* temporal interpolation: friend or foe ? responsible for ghosting? alternative?
semi-discrete optimization? mixed-integer programming? any reference?

* NNF is super slow (30min  with 6- neighborhood, 10min without interframe
regularization, for one pass, need 2)
* NNF time correspondence, seems messy useless, increases with time, matches anything,
especially BG. Too many good matches, (every frame) Larger patches ? 
* NN style correspondence seems to help with long-range in time. Misses some
details (time or space) because allows only 40 label with simple motion model,
globally -> do frame by frame? allow more models?
* NNF segmentation via graphcut -> superlinear in number of labels
* Alternative coarse temporal registration ? Discrete space-time interest points
 a la Harris corners?

 * We dont use a motion model for now.
 * Long distance matches still screwed up, especially time map

 * Result with/without mask, doesnt seem to change much -> background does not
 bias the minimum too strongly?
 * Result with/without decoupled regularization, seems to help time
 discontinuities especially

 * space time sift/ harris ? higher order derivs?

 FOV, occlusions, undersampling -> input videos
 * 3d ST affine ransac 4pairs pts

 2014-04-10
-----------

* 3D GCO is superslow ~2h, doesnt seem to help
* per frame 2D GCO ~ok 20min
* Overall NNF impact ~not super convincing
* try on coarser spatial resolution?
* Basket failure example

* For golfball -> other problem?
* Optical flow regularization number 2

* feature matching?
* Just got answer from Ivan Laptev: code for space-time interest points
* Try to get a coarse registration

* work on reconstruction
* Cleaner output warping field
* Better median filtering, cleaner discontinuities
* foward warping, poisson reconstruct, interpolation, use optical flow/ particle
videos for temporal processing? measure of how reliable?
* Use optical flow as a cue of who should be in front?

* get simpler videos

* add uncertainty measure to patch match
* prior on pyramid data term for mixing with NNF
* Keep track of multiple potential solutions?
* second nearest neighbor: 2nn test in sift and co: having two equally likely
candidates : mean probably a bad patch, covariance a little bit too permissive
* Explore more what 3d affine means, how it behaves, other models?
* Bias patchmatch, towards spacetime interst points using Laptev's code
* Patchmatch, limit deltaT +-20 frames
* Find patches with similar flow/incorporate flow metric in the matching/use
ST-HOG distance between patches
* Mix'em all

2014-04-16
----------

* Time regularization messes things up, but needed !

2014-04-23
----------

* GolfBall: light changes: contrast normalization ?
* Does the spatially varying time map allows for the appearance of objects twice
at different locations ? which would cause the wierd local optima?

2014-04-24
----------

* Mail to CSAIL for mindstorm or other robot
* Humming bird

* NNF usually doesnt find enough fitting affine transforms/homographies
 - not enough sampling?
 - wrong model?
 - useless NNF?
* Defaults back to translation only

* Debug on golfball
* Time regularization
* Time map meaningless? useless? 
* piecewise constant temporal term?
* Objects get sheared

* Ghosting and double object problem: two occurences in the cost 
 - Related to the long distance matching with NNF
 - upper bound on quality: cf. optical flow
 - can appear via time difference
 - can appear via C, warped video difference
 - can appear via frame interpolation (non-integer w)
 - would optical flow regularization solve that?
* Sand's video matching? frame to frame image registration + all pairs

We need to get closer to the optimum, coarse initialization
* Use the redundancy/correlation of video data
* patchmatch propagration using optical flow? different patchmatching metric?
* Seed a few patchmatch from interest points agreement
* coarse frame-frame matching using hog or sth, for time offset
* then coarse image registration for spatial offset
* use that as init

Friday afternoon:
* finish the optical flow based regularization

* Depends on initial frame/framecount?
* Objects have to follow a close trajectory (golfShort at the end, when one
    video didnt span this part of the motion)
* Initialization on frame one? this is the ref time/position?
* User input of reference keyframes to align the motion at certain positions, hard
constraints

* Separable initialization

Viewfinder alignment Andrew Adams
Image to Image alignment cheaply
Histogram of motion
Matching histogram of flow value in time
Regularize spatially based on optical flow similarity

DTW with optical flow
Gaussian falloff on the boundaries in time
Make it work for different video lengths: boundary conditions screwed up might
propagate?

2014-05-12 - return from NYC
----------

+ Implemented flow driven regularization/neighborhood system
* Flow driven regularization does not seem to provide additional benefits
+ Enabled different video lengths
* Boundary conditions seem to matter a lot: having the same beginning and end
keyframe helps -> closer to a good solution (temporally)
* Initial temporal registration should help
* golfBall: rotation gives a timing constraint, then translation is messed up,
    non uniform across the ball
- Try input a manual warping/time map to bring objects close together and see if
it is works getting close to the optimum

2014-05-14
----------

* Global flow histograms: not very informative (event to DTW?)
* Different pyramid schedule: time on lowres, then upsample spatially: no
difference it seems
* Manual initialization: seems to help. The backward warp is quasi-perfect, ie
cost is minimized. However, is the map meaningful? rigid object deform
(newtonBall)
- Try a uniform mapping index?
* How to leverage space-time interest points?
* flow as data term is useless, too smooth, not discriminative enough

2014-05-15
----------

* Is it ok to have some user input? A few synchronization points?
* Smooth time map?

2014-05-16
----------

--- TRIED ---
* Videos of different lengths, initialize with linear time mapping
* Both inputs depict the same spatial extent of the motion.

* Flow in the data term messes things up
* Manual time warping on newton

* Flow driven regularization, seemed marginal on golf an newton example

--- DECOUPLING TIME FROM SPACE alignment ---
* hope to get meaningful correspondence of 'events', instead of just match
whatever is cheaper
* Different pyramid schedule: time uspampling first on spatial-lowres, then
upsample spatially: havent dug enough to see a difference

--- GET A coarse time alignment ---
* Dont want to track, or manually specifiy an object
* Build a feature vector for each frame, then apply Dynamic Time Warping
* Assume same ordering of events in inputs: monotonic mapping
* Global flow histogram: useless
* Global histogram of flow orientation, weighted by magnitude: not much better
than random features
* Spatial histogram of flow, weighted by magnitude, only on one axis: better
- Try spatial histogram of flow orientation, weighted by magnitude?
normalization? make it scale invariant?
- What would be a good representation? Should we make more assumptions?
Spatially close?

* Spatially varying time map, breaks the rigidity of objects:
    - uniform time map?
    - smooth time map (ie dont allow discontinuities?)
    - regularize time map based on flow similarity? Objects that move together
    should be time-mapped together?

--- What else ---
* A few videos from sports (baseball highjump, not processed yet): pb camera,
    public motion
* User input?

--- Rendering ---
* Layer decomposition?
* Forward warping only is not enough: compare alignment with forward and
backwarp warping only
* Clueless about layer ordering.
* Introduces noise, and small oscillations by rounding off to the nearest pixel
* How to measure overlap?

* overlapped bins, tent weighting -> avoid aliasing
* regularization in the DTW

2014-05-23 - meeting with Fredo
----------

General:
* Not very confident on the outcome of the current method
* Energy is close to be minimized, difference image shows pixel error is low.
* But correspondence does not seem always reasonable
* Amplificating translational offset poses several problems:
    - layer ordering: depth would help, but changes the scope
    - complex motion synthesis when amplification is large: local transform
    estimation? segment, graph constrained optimization?

On time alignment:
* DTW, essentially discrete, 
* global histogram of flow: does not seem discriminative enough between frames
* re-introduce spatial cues: spatial grid, local histogram of flow, normalize
wrt to magnitude, weight by magnitude, not much help, soft binning
(interpolation) to avoid aliasing. Better, but still so-so.
* Somewhat working on Newton, but not on derailleur, golfBall, walk,
others
* bag-of-words approach (try to get rid of spatial grid):
    - superpixels segmentation (better than just a grid?)
    - flow + color + gradient feature
    - kmeans clustering, gives a dictionary
    - histogram of word count
    - same as space alignment
* Definition on the temporal alignment? Point in time where the trajectories are
closest? semantic defintion?

Misc:
* tried to change the schedule of the coarse-to-fine:
  perform temporale upsampling only on the coarse spatial version, then
  spatial
* tried not optimizing for time offset (ie global from DTW), then frame to frame
offset computation with time regularization
* Errors in the warp-field, bother amplification as much. (same with temporally
unconstrained frame to frame optical flow)

Optimization does it's job, but what's the meaning of this correspondence?
somewhat arbitrary.
Should we refocus the scope of the project? is it hopeless?
Ways to use the mapping less directly for amplification?

* Intra-video correspondence, motion estimation
* Inter-video correspondence in space and time

* regularization dependent on qtity of movement in video

* try add regul in the dtw cost, cubic interpolation -> deal with discrete
