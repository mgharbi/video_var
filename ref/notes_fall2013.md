Motion Difference Magnification
===============================
Michael Gharbi <gharbi@mit.edu>

2013


2013-05-02
----------
The goal of the project is to compare similar motions taken from two input
video streams and amplify differences that could be barely visible under the naked
eye.

One can seek to compare the swing of Tiger Woods and any other professional golf
player to learn what subtle motion gives the master an edge over its
challengers. How consistent are Nadal's services and why does he miss some of
them?

* Motivating examples:
    - Golf player
    - Tennis player
    - Drummer => high speed video
    - Heat propagation, dilatation vs. several physical conditions/materials
    - Surgery => precision

* Applications:
    - Motion derivative ?
    - Motion difference with same input but offset?

* Experimental validation:

* Foreseen obstacles:
    - precise registration of human originated motions

* Brainstorm:
    - What to amplify: human actions, mechanical machines?
    - Why would we want to visually amplify motion difference: diagnostic tool? reveal?
    EM microscopy? Biology cultures 


2013-05-13 - Meeting with Miki Rubinstein
----------
Visualization is a problem of its own.

In typical examples (car running over an obstacle, roquettes), the difference is
on a scale way above the acceptable range for Eulerian Motion Magnification.
Therefore the output is very noisy. Motion attenuation works better in these
cases (Taylor expansion is a more acceptable approximation in these range).

Build on a motion model and compare differences to the model. They will
hopefully be low, and thus in the range of the method. Can be used for quality
control in industrial setups. Balance between the full motion registration of
the Lagrangian approach and the Eulerian one. 

Kalman filtering

* Talk to Ce Liu
* Non chronological video synopsis, Schmuel Peleg


2013-05-24 - Meeting with Sylvain and Fredo
----------
Motion magnification for timelapses and slow-motion. Attenuation generally gives
better results than amplification.

For timelapses:

* removing trees wobbling motion
* removing pedestrians
* rendering a more natural motion

Define a target optical flow (or constrain it) e.g. the optical flow of a video
at normal speed. Use image-based priors: the output should look like a 'real'
piece of video. Freezing motions, smoothing others (pedestrian, not jittering
around). The goal is to have a user-prescribed optical flow.

Use of high-framerate camera, or video-based timelapses: how to play at
different speed while preserving motion details. Discard frames is ok since
there is plenty of data.

Start with an ideal input: full video recording of 24h. Then see how to
relax/degrade: short video sequences every 10 minutes?

Combining gracefully large and small scale changes (e.g building construction
site, vs. workers wandering around). Temporal editing, control the optical flow
statistics.

Example videos to look at:

* NASA timelapses
* Cheetah running (National Geo)
* Salamander: growing

References:

* John Fisher, Learning Visual Flows: A lie Algebraic Approach
* Eric P. Bennett & Leonard McMillan

Slow-motion:

* check Twixtor, After Effects
* motion denoising
* minify changes
* pathcMatch like registration of close frames. Can't map full frames =>
locality


2013-06-05
----------

Abe: video microphone


2013-06-12
----------

MRI images, medical images. Embryo developpement. Cell growth. Comparison to a
model, measure of deviation + amplification.


2013-06-17
----------

Image registration: align by hand ?
Large scale vs. small scale motions.

* Align large scale motions
* Filter small scale motion
* Amplify


2013-07-12
----------

Pipeline:

* space-time align video 1 and 2 => warping function
* compute optical flow (for each? between the two aligned streams?)
* filter scale of differences in the warped streams: e.g. fix large scale
differences (another warp?), and amplify tiny ones.
* amplify a desired band


2013-07-15
----------

Learning by comparing motion -> link to Alec Rivers' work.


2013-07-16 - Meeting with Fredo
----------

Motion Capture:

* Dynamic Time Warping (speech, retiming)
* motion graph
* *styles* (walks and co)

Different inputs: (2d will be a problem)

* kinect
* mocap

From biomechanic litterature:

* walk
* *symmetries*

Input videos, models:

* vibration models: changing the materials, adding objects on the moving part
etc
* physical, repetitive, mechanical motions

Application -> analyze *spin* in golf, tennis (e.g. subtract parabolic motion)


2013-08-06
----------

*Canonical Time Warping* & *Generalized Time Warping* Zhou et al.
Builds on *Dynamic Time Warping* and Canonical Correlation Analysis
Integer alignment of time series.

Video features?

Weizmann database for human motions


2013-08-12
----------

Read Abe's lightfield and YiChang laser speckles.


2013-08-15
----------

Cen Rao *View-invariant Alignment and Matching of Video Sequences* suggests
application to training dancers.

Look at local freeform registration, warping.


2013-08-19
----------

Have a look at symmetry: compare walk to detect assymetries.

Trajectory-based temporal registration:

* What about different temporal scales for different parts of the motion (e.g.
different fingers when playing piano).

* Which difference to work on: optical flow? Pixel to pixel is meaningless when
input videos are from completely different scenes.


2013-08-21
----------

Bounding the optical flow, with a threshold T:
* if magnitude < T, use the truncated flow for warping
* if magnitude > T, use magnitude-T for amplification


2013-08-26
----------

UCF sports dataset

Maybe extend CTW or other DTW to take into account multiple trajectories, and
find a balanced solution that best time-warp the global sequence.


2013-08-28
----------

Nadal video:
* temporal registration is ok
* space registration/warping remains to do
* background and object are the same


Fingers video:
* temporal registration remains to do
* space registration is ok
* background and object are the same


2013-08-29
----------

* Talk to a concert pianist
* Walk symmetries
* Surgeon
* Gifford and Boston Ballet

Other direction: population analysis (e.g. Nadal's serves), classify which goes to the net, which
scores, etc


2013-08-30 - Meeting with Fredo and Bill
----------

* Need to clarify the goal and application
* Unclear who's going to use it
* Unclear what is the contribution, are we making the problem artificially hard?

* Need to confirm usability with chiropodist, surgeon, pianist, dancer,
tennis player...

* Goal: match videos of similar actions, performed by different persons in
different contexts.
* What else can we use motion comparisons for? Industrial applications, stress
analysis, speed variation in chain production.
* Assumptions: static scene, one object of interest
* Draft idea of a pipeline

* Other directions (more vision style):
    * Population analysis through motion
    * Tennis successful serves
    * Where a football player positions himself, and where does he shoot

-- notes

* keboard typing as a biometric: people have their very unique way of typing 
* way people stand as a biometric
* civil engineering people stability of structures (Shell Gas and Oil platform)
    Vibrations on a platform rig before and after a damage
    vs. acoustic power spectrum, vibration modes

* Difference between spatio temporal difference in freq domain
What's the source of motion

Justin postdoc?

Biomechanical litt, study of people walking through motion cap

Bike - true/ untrue wheel.
Structural defects

Where does seeing better small motion differences matters?

Capture a walk with/without backpack/pebble in shoe
Capture piano through mirror
Capture hitting metal rod before/after dammage
Capture bike wheel true/untrue, flat tire

Shell Oil vibration modes

Sylvain suggests looking at MIT diving team data


2013-09-03
----------

Contact people:

* Divers at MIT : snodgras@mit.edu - Brad Snodgrass

* Piano teacher, or concertist
* Dancers at the Boston Ballet
* Architecture, structural analysis, acoustic, vibration modes: Jerome J. Connor
* Chiropodist: Brittany, Brian Ash, J.Y. Cornu

Data:
* Walk and run on a treadmill at MIT
* Piano play with mirror


2013-09-04
----------

Z-center should contact for video authorization

Capture walk cycles @ Z-center


2013-09-05
----------

* Hugh Herr
<http://www.youtube.com/watch?v=8AoRmlAZVTs>
<http://www.media.mit.edu/people/hherr>

* Farish Jenkins
High speed cineradiography, Wind tunnel data

chiropodist: ask Jean Langlois from Boston French Connection


2013-09-06
----------

Two main concerns:
* who is going to use that? what for?
* what do we amplify? how do we visualize?

Make synthetic data

Quantitave deviation from a model?

Could be used as a validation method for the success of Hugh Herr's prosthetics,
comparison to the gait of a non handicapped person


2013-09-16
----------

Difference amplification not entirely satisfying. Need for
regularization/smoothing. 

What happens for circular trajectories?

Use SimpleFlow for fast OpticalFlow (M.Tao and J.Bai)


2013-09-17
----------

* Use flow discontinuities to align/warp images
* noise and high frequency disturb visualization
* calculate flow A->B to use as a warp and better align images when comparing
the flow


2013-09-18 - Group meeting
----------

* Flood fill/ boundary fill to extract silhouette from flow
* Background subtraction for better composite

Valentina:
* Look at Jessica Hodgins's work from CMU: <http://mocap.cs.cmu.edu>

* Synth example: parabolic traj + sin oscillation
* difference of time derivative between corresponding points
* tried to apply to real videos:
    - compute flow
    - warp B to A
    - compute difference of flows
    - amplify
    - add back

Inaccurate correspondences, noisy flow


2013-09-20 - Meeting with Fredo
----------

* Next: simpler cg model, mass-spring dynamic simulation for more
predictable/expected results

* Pb: inaccurate registration, blocky optical flow, maybe a global optimization
some time in the future (like Irani's video completion)

Questioning:
* What happens in Fourier domain?
* What to do with space vs time dimension?
* Big least square optimization: amplify speed by alpha, while mainting
coherence etc?

Examples:
* Simple example earth, moon, sun, 3body problem with or without the moon: amplify
the differences.
* Tire balance: wheels not properly aligned (weight on car wheels) 
* Stable fluids
* vibration, simulations of complex spatial motions 

Optical flow: Ce Liu

Talk to David Levin, see if he has some physics simulation code in handy.


2013-09-23
----------

* Mail to Michael Rubinstein to join the next meeting with civil engineering people.
* Mail to Hugh Herr for gait data.

* Mail to David Leving for physics simulation code.
    - Bullet Physics Library - Rigid bodies
    - VEFA FEM - deformable bodies


2013-09-25
----------

* Maya scene
* Look at Ce Liu, assisted optical flow


2013-09-30
----------

* Motion segmentation
* Multiframes optical flow, temporal coherence
* problems with inverse warping
* Read original motion mag paper for motion segmentation and warping


2013-10-01
----------

* Sand and Teller 2004 - Video Matching
* Sand and Teller 2006 - Particle Video
* Yizhou Yu 2004 - Video Metamorphosis Using Dense Flow Fields


2013-10-02 - Group Meeting
----------

* Reconstruction first: "what if we have a perfect displacement field"
* Then temporal/spatial regularization/optimization
* Rethink the definition of displacement field

For now: match velocity fields

* Debug optical flow: warp from frame1

2013-10-04 - Meeting with Fredo
----------

* Optical flow fixed with Ce Liu's code. Less accurate near border, but faster
and smooth

* Problem with forward warping (vs. reverse): original motion mag compositing?
    -
* Definition of what we want to amplify: pos (DC component), velocity difference
    -

* Making slides

* Attenuation of the velocity perpendicular to direction of most variation?

* Why velocity?
    - DC component for position matching, define? spatially varying, not just
    barycenter
    - temporal filtering: get difference frequencies ampl
* Why no integration?
    - error propagation

    Visualization question: edge detection on one + superimpose


2013-10-09 - Feedbacks from CGGAR
---------

* Wojciech: Control to balance between aligning and pushing apart the videos
* YiChang: Arrows for visualization instead of warping
* Valentina: Amplify only one video compared to the original other
* Jonathan: mechanical examples. Guitar string, Dslr vibration example
* Controls for what we want to amplify: velocity, position
* Sylvain: Push things in the temporal domain, instead of spatial
* Ariel Shamir: cluster similar motions and warp them together instead of local per-pixel
optical flow. For music, audio output from the time-warped videos. Or audio to
align?


2013-10-10
----------

* Switch to feature tracking? instead of optical flow
then Optical flow interpolation like original motion mag paper?
* user input for the points/object to track? (segmentation a la photoshop magic wand, vs background)
* what kind of time processing? discrete features -> DTW
* Motion clustering to remove flicker and artefacts?

* CG example spinning cube track corners
* Mass spring systems
* Visualization: lines overlaid with original, compare to original, compare
A-amplified to B-original


2013-10-11 - Meeting with Fredo
----------

* For Justin's high speed data: random excitation, work with power spectrum, amphasize difference of spectrum, random sig 
freq processing
* look for better optical flow
* Play with phased-based code

Video is hard to visualize:
* A vs average of A and B, difference to the baseline
* Amplified diff btwn color channels, image gradients, edge detection on one image
* increase saturation or other color processing between channels
* heatmap of where the motion differ
* Leave a trace along the trajectory of points that are tangent to the motion

2013-10-18 - Meeting with Fredo
----------

* Stuck on Formalization/Formulation of the problem
* Mathematical formulation, invariants?
* Rigid vs. non rigid. Amplify transformation?
* Justification? What makes sense? What is the invariant we want to preserve.
* If video visualization is hard, can we come up with a measurement of the
motion difference?
* Reconstructing motion on a static frame?

References - Local rigid transformation:
* Transform Matrix decomposition: polar decomposition
* As rigid as posible shape manipulation - Takeo Igarashi
* image deformation using moving least squares - Scott Schaefer
* ilya Baran - Bounded biharmonic weights for real time transformation

Dealing with occlusion : Ce Liu MotionMag
more advanced models of local appearance for feat tracking
modern tracking algo : flow + classification : learn classifier for local

Optimize using both videos simultaneously to get the motion and the time warping
appearance
Formulate as a big optimization with sparse features tracked.
Opt Flow - Sift - Global optimization for time warping + regul over velocity
field and time warping 
use ADMM?

Moving least squares.


2013-10-25 - Meeting with Fredo
----------

Model:
* sparse set of features
* triangulations to get more than point translations. Is it a gd approach?
* Skipped the motion estimation + time warping joint optimization for now

* How to process the motion when we have this info?
* estimate transforms for each triangle in both videos
* blend transforms of corresponding triangles -> how? rigid motion? what to
preserve? how to amplify rotation+translation
* Marc Alexa, Ken Shoemake (Linear combinations of transformations, Polar
decomposition)
* Is it the right path? Should we stick to translations only? Model by
rigid components? (triangle with low deformation are linked together and follow
the same transform, instead of per triangle, weighted graph for strength of
connection)
* Energy formulation?

Idea for visualization:
* Start from reference frame, and warp it using the 'difference transforms'
only.

* selective deanimation paper
* scale down the general motion instead of completely removing it
* change of coordinates to track the global motion
* Kinematic model for fingers


2013-10-30
----------

* triangulation based to capture rigid motion (point translation only causes
homothetic transforms)
* M.Alexa linear combination of transforms + polar decomposition : Limited
amplification range.
* Challenge of visualizing the differences : static pose transformed
* Discrete feature -> Tracking is a weak link
* Time warping + time processing, amplify differences in timing


2013-11-01 - Meeting with Fredo
----------

* Exponential map for transforms give reasonable results, but not entirely
satisfying
* For rotation, what a reasonable amplification ? Maybe a shift in time domain
is more appropriate ?
* Tracking is painful
* Trying with particle advection on the optical flow field (inspired from
crowd tracking): gives trajectory,
useful for time warping, classification, matching?
* Using short pieces of trajectories to classify/infer motion type (rotation,
translation) and amplify accordingly
* Lie Algebraic approach to visual flow paper
* In the case of global optimization : spatially varying time warping/correspondence map?
3-coordinates optical flow?

Preprocess for stability.
Video Mesh project jiawen chen
kristen grauman
active contours / snakes
multiscale 
occlusion handling > check middlebury
magic hand scissor/ user input

maneesh agrawala - soft scissors
jue wang - towards temporally cohering video matting

refs:
<http://vision.cs.utexas.edu/projects/bplr/>
<http://vis.berkeley.edu/papers/softscissors/>
<http://juew.org/publication/Mirage11-videoMatting.pdf>


2013-11-08 - Meeting with Fredo
----------

James & twig *Skinning Mesh Animation*
* extract bones and rigid components, might be useful for animation

Eric Berson - Orthopaedic Surgery: biomechanical analysis and motion analysis
<eberkson@mgh.harvard.edu>

Good quality motion estimation... Compare motions without estimating them
precisely? 

Need a confidence measurement to discard meaning less correspondences

Motion estimation. Joint estimation using both video?
Correspondence
Segmentation?
Time correspondence from dense data

* Write down christmas list of features we want our matching to have -> energy to optimize
* Energy formulation: space time patches
* (x_a,y_a,t_a)< - > (x_b,y_b,t_b) correspondences. What happens with fractional
t's? What do we do with spatially varying time warp? Force timing to be the
same/close spatially? Things worth precomputing: optical flow?
* Start with one point trajectory
* Ce Liu motionmag, solving for the mask at the same time, co-segmentation. How
do they deal with occlusion.
* Ask Ce Liu for litterature on joint solving of optical flow/time warp
* Guha basket video
* Start with artificial data, stretch video and see. Start experimenting with a single trajectory
* Fit local rigid motion, moving robust least square: eliminate points that dont
match the motion model. Pruning outliers


2013-11-13
----------

I have another name for you!  Sorry for taking so long on this one.  I initially
emailed an ortho sports medicine fellow and never heard back from him (he's just
super busy).  I eventually contacted my friend Steve Southard who does
sports medicine (but from the medicine perspective, not the ortho perspective).

He's recommending that you contact Eric Berson at eberkson@mgh.harvard.edu.  His bio is at: 
<http://www.massgeneral.org/ortho/doctors/doctor.aspx?id=17819>

Research profile:
<http://connects.catalyst.harvard.edu/Profiles/display/Person/33556>


2013-11-14
----------

* Space-time registration is a problem in itself:
    - nothing in the litterature (?) about 'exact' space-time registration
* Most methods estimate a global space-time (homography+affine) transform (caspi, irani)
* Others that enable subpixel/subframe accuracy but work on (generally single
pair of) trajectories (CTW,DTW)
* quantitative motion comparison: delay, spatial offset directly encoded (even if no visual amplification)
* Use local motion features only? Or do we need global trajectories? What's the
richest representation?
* Will this free us from having an explicit motion model?
* Use patch correlation as metric (appearance independent? Irani et al.)
* Dense/sparse?
* Balance space/time
* Occlusion? (not for now)
* initial warp? (eg SIFTflow)

Global ST-patches search:
* Lucas-Kanade style in 3d? but no global transform
+ no explicit motion model
- where to look for the best matches? space-time neighborhood? How to constrain it?
- tractable?
- what to compare? what is the motion information?
- is patch to patch insightful? rich enough of a model to capture motion
- eulerian processing? displacement u(x,y,t) ?

Sparse feature based:
+ compute a 3D ST-warp

Trajectory based:
- needs full motion estimation
- requires long range tracks
- how to match corresponding features
+ can use KLT, optical flow, particles video

Whish list:
* handle non-trivial, non-parametric time warps
* fractional/continuous time and space (non-integer, interpolated)
* preserve causality (monotonic in t)
* injective warp, no fold (Feng Liu, Content-Preserving Warps for 3D video
stabilization), monotonous?
* smooth warp (space and time)
* cost on spatial and temporal distance:
prefer closer patches.

* Try brute force correspondences computation (scanning window style)
* Capture turning bike wheel
* Capture finger motion, of 2 persons, with different backgrounds


2013-11-15 - Meeting with Fredo
----------

Generalization of patch: one u,v,t for each pixel Yael Pritch Shiftmap Editing -> interesting representation
Start with identity mapping and move from there (deform)

SpaceTime patches, Lucas Kanade

Still problem with occlusion
Use optical flow in the features space


2013-11-15 - Meeting with Amun Makani
----------
Amun Makani phone: 304-668-0051


2013-12-04
----------

General Construction of Time-Domain Filters for Orientation Data
Jehee Lee
http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=998665


2013-12-06 - Meeting with Fredo
----------

* Implemented ST Pyramid with anisotropic Gaussian Blur in 3D
* No further spatial refinement yet
* Havent play with params
* Much better results (visually), energy decreases (sanity check)
* Iterative refinement
* Still quite some noise (similar to Black's HS implementation)

* Constraints on the jacobian for monotonicity/injectivity
* Different data term, metric, to take into account videos with different setups
* Debug the warping phase

Feedbacks:
* Implement known goodies from optical flow methods. 
  Technical difficulties: different videos, correlation, good metric, new metric,
  learn a mapping.
  Either similar: such that optical flow works anyway (or stereo), then later
  generalize
  Examples with symmetry timing (people rowing, cowboys drawing guns)
* Experiment with ST regularization, various constraints, color gradients etc
* Worry about Size/memory and perfs?
* Fix the warping issue (double ghosts)
* What can be done with the Jacobian?
* Find convincing examples to show the time offsets
  Mechanical systems, Ce, Michael, Car with something in the trunk driving over a
  bump.
  ski turning, timing of the slowdown, speedup, jumps
  fatigue effects, success/failure amplification
  Guha, basket tennis, other sports
* Try again the amplification


2013-12-09
----------

* Write a visualization function that has a global scaling for the flow
* Memory footprint is insanely huge for 405x720x76 videos. The linear system
takes a while to be solved.

* Add time coordinate scaling (x, y, ct)
* Implement robust functions for data and smoothness
* Add spatial weight on the regularization

* How to bias solution towards a unique global timewarp per frame ?
* How to enforce monotonicity?
* Understand space-time interplay, how to balance prederence?
* How a decoupled search vs. gloabal opt would behave?

* Time resolution is too low, variations are too big

Local correlation measures for motion analysis


2013-12-12
----------

* We might want to use multigrid instead of multiresolution for speed and
optimality
* CG data for the evaluation of 3D flow, ~middlebury ?
* Or application examples ?

2013-12-13 - Meeting with Fredo
----------

* Bug in the filters
* Debugging robust implementation
* Numerical scheme for linearization: FP iterations, probably SOR 
* Memory is close to be limiting
* Can we do a tiling scheme and propagate from boundary conditions?

* Other constraints in the energy formulation?
* How to use the warping field to amplify motion differences ?

Layering
Write draft for pre-deadline
Tinkering with details
Play with ST regularization
What to compare to? Irani space-time. 
Motion-Comparison good motivating examples
Compare to magnification of space alone: that give insights to people

Neal's code phase-based

Lagragian in time, Eulerian in space alignment ? 1d time warping with Normalize
Cross Correlation -> Future work

Memory, time, Coarse to fine: smart scheduling, number of iteration per level
Use BigBoy

We might need some layering

2013-12-16
----------

* Refine spatial pyramid
* Influence of time scale?
* Problem of noise in the input ?
* Robust data term is broken: numerical issues? Just invalid?
* Weighted median filter
* A is singular !
* Tiling scheme to fit in memory?
* Use a smoothness matrix to control u,v space/time regularization, and v
space/time regularization, all independently

* System badly conditioned
* Numerical issues, conjuguate gradient, SoR fail to converge or numerical
issues (scalar out of range)
* general min residual, better, still doesnt converge
* Tried Cholesky, LU, Jacobi preconditioner, not much help here
* Typical size 20e6, non zero elements ~1e6
* Even worse with robust, non linear cost functions

* Add a part on decomposing with boundary conditions
* Add 0.45 generalized charbonnier - done
* Improved median filtering
* Implement finer spatial pyramid
* Pyramid should have different scaling factors for time ? ie more progressive
in time
* Boundary conditions

2013-12-17
----------

* Fixed A

* Find convincing examples ! Where motion difference is subtle

2013-12-18
----------

* Filters/ structure-texture decomposition/ median filtering

2013-12-20 - Meeting with Fredo
----------

* vary robustness functions
* Integer time warping?
* Force global map?

Create synthetic examples for proof of success

Intro 
    diff in position, velocity, sometimes just timing matters
    synthetic data

Illumination variation in Rod movie

2013-12-23
----------

Solving the linear system too well causes the noisy artifacts (pcg ILU or
backslash). This is wierd, Ce Liu's code does not have this problem.

