       ЃK"	  Рћ
жAbrain.Event:2ѕЏЅ1      я_z	nыћ
жA"c
Y
x-inputPlaceholder*
dtype0*
shape: *'
_output_shapes
:џџџџџџџџџ!
W
hrandPlaceholder*
dtype0*
shape: *'
_output_shapes
:џџџџџџџџџ(
W
vrandPlaceholder*
dtype0*
shape: *'
_output_shapes
:џџџџџџџџџ!
N
PlaceholderPlaceholder*
dtype0*
shape: *
_output_shapes
:
M

keep-probsPlaceholder*
dtype0*
shape: *
_output_shapes
:
g
truncated_normal/shapeConst*
dtype0*
valueB"!   (   *
_output_shapes
:
Z
truncated_normal/meanConst*
dtype0*
valueB
 *    *
_output_shapes
: 
\
truncated_normal/stddevConst*
dtype0*
valueB
 *ЭЬЬ=*
_output_shapes
: 

 truncated_normal/TruncatedNormalTruncatedNormaltruncated_normal/shape*
dtype0*
seed2 *

seed *
T0*
_output_shapes

:!(

truncated_normal/mulMul truncated_normal/TruncatedNormaltruncated_normal/stddev*
T0*
_output_shapes

:!(
m
truncated_normalAddtruncated_normal/multruncated_normal/mean*
T0*
_output_shapes

:!(
y
weightsVariable*
dtype0*
shape
:!(*
	container *
shared_name *
_output_shapes

:!(
Ё
weights/AssignAssignweightstruncated_normal*
validate_shape(*
_class
loc:@weights*
use_locking(*
T0*
_output_shapes

:!(
f
weights/readIdentityweights*
_class
loc:@weights*
T0*
_output_shapes

:!(
R
ConstConst*
dtype0*
valueB(*ЭЬЬ=*
_output_shapes
:(
u
hidden-biasVariable*
dtype0*
shape:(*
	container *
shared_name *
_output_shapes
:(

hidden-bias/AssignAssignhidden-biasConst*
validate_shape(*
_class
loc:@hidden-bias*
use_locking(*
T0*
_output_shapes
:(
n
hidden-bias/readIdentityhidden-bias*
_class
loc:@hidden-bias*
T0*
_output_shapes
:(
T
Const_1Const*
dtype0*
valueB!*ЭЬЬ=*
_output_shapes
:!
v
visible-biasVariable*
dtype0*
shape:!*
	container *
shared_name *
_output_shapes
:!
Ѓ
visible-bias/AssignAssignvisible-biasConst_1*
validate_shape(*
_class
loc:@visible-bias*
use_locking(*
T0*
_output_shapes
:!
q
visible-bias/readIdentityvisible-bias*
_class
loc:@visible-bias*
T0*
_output_shapes
:!

MatMulMatMulx-inputweights/read*
transpose_b( *
transpose_a( *
T0*'
_output_shapes
:џџџџџџџџџ(
V
AddAddMatMulhidden-bias/read*
T0*'
_output_shapes
:џџџџџџџџџ(
I
SigmoidSigmoidAdd*
T0*'
_output_shapes
:џџџџџџџџџ(
L
subSubSigmoidhrand*
T0*'
_output_shapes
:џџџџџџџџџ(
C
SignSignsub*
T0*'
_output_shapes
:џџџџџџџџџ(
D
ReluReluSign*
T0*'
_output_shapes
:џџџџџџџџџ(
E
transpose/RankRankweights/read*
T0*
_output_shapes
: 
Q
transpose/sub/yConst*
dtype0*
value	B :*
_output_shapes
: 
V
transpose/subSubtranspose/Ranktranspose/sub/y*
T0*
_output_shapes
: 
W
transpose/Range/startConst*
dtype0*
value	B : *
_output_shapes
: 
W
transpose/Range/deltaConst*
dtype0*
value	B :*
_output_shapes
: 
~
transpose/RangeRangetranspose/Range/starttranspose/Ranktranspose/Range/delta*

Tidx0*
_output_shapes
:
[
transpose/sub_1Subtranspose/subtranspose/Range*
T0*
_output_shapes
:
k
	transpose	Transposeweights/readtranspose/sub_1*
Tperm0*
T0*
_output_shapes

:(!
~
MatMul_1MatMulSigmoid	transpose*
transpose_b( *
transpose_a( *
T0*'
_output_shapes
:џџџџџџџџџ!
[
Add_1AddMatMul_1visible-bias/read*
T0*'
_output_shapes
:џџџџџџџџџ!
M
	Sigmoid_1SigmoidAdd_1*
T0*'
_output_shapes
:џџџџџџџџџ!

MatMul_2MatMulx-inputweights/read*
transpose_b( *
transpose_a( *
T0*'
_output_shapes
:џџџџџџџџџ(
Z
Add_2AddMatMul_2hidden-bias/read*
T0*'
_output_shapes
:џџџџџџџџџ(
M
	Sigmoid_2SigmoidAdd_2*
T0*'
_output_shapes
:џџџџџџџџџ(
P
sub_1Sub	Sigmoid_2hrand*
T0*'
_output_shapes
:џџџџџџџџџ(
G
Sign_1Signsub_1*
T0*'
_output_shapes
:џџџџџџџџџ(
H
Relu_1ReluSign_1*
T0*'
_output_shapes
:џџџџџџџџџ(
G
transpose_1/RankRankweights/read*
T0*
_output_shapes
: 
S
transpose_1/sub/yConst*
dtype0*
value	B :*
_output_shapes
: 
\
transpose_1/subSubtranspose_1/Ranktranspose_1/sub/y*
T0*
_output_shapes
: 
Y
transpose_1/Range/startConst*
dtype0*
value	B : *
_output_shapes
: 
Y
transpose_1/Range/deltaConst*
dtype0*
value	B :*
_output_shapes
: 

transpose_1/RangeRangetranspose_1/Range/starttranspose_1/Ranktranspose_1/Range/delta*

Tidx0*
_output_shapes
:
a
transpose_1/sub_1Subtranspose_1/subtranspose_1/Range*
T0*
_output_shapes
:
o
transpose_1	Transposeweights/readtranspose_1/sub_1*
Tperm0*
T0*
_output_shapes

:(!

MatMul_3MatMul	Sigmoid_2transpose_1*
transpose_b( *
transpose_a( *
T0*'
_output_shapes
:џџџџџџџџџ!
[
Add_3AddMatMul_3visible-bias/read*
T0*'
_output_shapes
:џџџџџџџџџ!
M
	Sigmoid_3SigmoidAdd_3*
T0*'
_output_shapes
:џџџџџџџџџ!

MatMul_4MatMul	Sigmoid_3weights/read*
transpose_b( *
transpose_a( *
T0*'
_output_shapes
:џџџџџџџџџ(
Z
Add_4AddMatMul_4hidden-bias/read*
T0*'
_output_shapes
:џџџџџџџџџ(
M
	Sigmoid_4SigmoidAdd_4*
T0*'
_output_shapes
:џџџџџџџџџ(
P
sub_2Sub	Sigmoid_4hrand*
T0*'
_output_shapes
:џџџџџџџџџ(
G
Sign_2Signsub_2*
T0*'
_output_shapes
:џџџџџџџџџ(
H
Relu_2ReluSign_2*
T0*'
_output_shapes
:џџџџџџџџџ(
B
transpose_2/RankRankx-input*
T0*
_output_shapes
: 
S
transpose_2/sub/yConst*
dtype0*
value	B :*
_output_shapes
: 
\
transpose_2/subSubtranspose_2/Ranktranspose_2/sub/y*
T0*
_output_shapes
: 
Y
transpose_2/Range/startConst*
dtype0*
value	B : *
_output_shapes
: 
Y
transpose_2/Range/deltaConst*
dtype0*
value	B :*
_output_shapes
: 

transpose_2/RangeRangetranspose_2/Range/starttranspose_2/Ranktranspose_2/Range/delta*

Tidx0*
_output_shapes
:
a
transpose_2/sub_1Subtranspose_2/subtranspose_2/Range*
T0*
_output_shapes
:
s
transpose_2	Transposex-inputtranspose_2/sub_1*
Tperm0*
T0*'
_output_shapes
:!џџџџџџџџџ
v
MatMul_5MatMultranspose_2Relu_1*
transpose_b( *
transpose_a( *
T0*
_output_shapes

:!(
D
transpose_3/RankRank	Sigmoid_3*
T0*
_output_shapes
: 
S
transpose_3/sub/yConst*
dtype0*
value	B :*
_output_shapes
: 
\
transpose_3/subSubtranspose_3/Ranktranspose_3/sub/y*
T0*
_output_shapes
: 
Y
transpose_3/Range/startConst*
dtype0*
value	B : *
_output_shapes
: 
Y
transpose_3/Range/deltaConst*
dtype0*
value	B :*
_output_shapes
: 

transpose_3/RangeRangetranspose_3/Range/starttranspose_3/Ranktranspose_3/Range/delta*

Tidx0*
_output_shapes
:
a
transpose_3/sub_1Subtranspose_3/subtranspose_3/Range*
T0*
_output_shapes
:
u
transpose_3	Transpose	Sigmoid_3transpose_3/sub_1*
Tperm0*
T0*'
_output_shapes
:!џџџџџџџџџ
y
MatMul_6MatMultranspose_3	Sigmoid_4*
transpose_b( *
transpose_a( *
T0*
_output_shapes

:!(
I
sub_3SubMatMul_5MatMul_6*
T0*
_output_shapes

:!(
J
mul/xConst*
dtype0*
valueB
 *
з#<*
_output_shapes
: 
A
mulMulmul/xsub_3*
T0*
_output_shapes

:!(
N
	truediv/yConst*
dtype0*
valueB
 *   A*
_output_shapes
: 
G
truedivDivmul	truediv/y*
T0*
_output_shapes

:!(

	AssignAdd	AssignAddweightstruediv*
_class
loc:@weights*
use_locking( *
T0*
_output_shapes

:!(
R
SubSub	Sigmoid_2	Sigmoid_4*
T0*'
_output_shapes
:џџџџџџџџџ(
X
Mean/reduction_indicesConst*
dtype0*
value	B : *
_output_shapes
: 
k
MeanMeanSubMean/reduction_indices*
_output_shapes
:(*
T0*
	keep_dims( *

Tidx0
J
Mul/xConst*
dtype0*
valueB
 *
з#<*
_output_shapes
: 
<
MulMulMul/xMean*
T0*
_output_shapes
:(

AssignAdd_1	AssignAddhidden-biasMul*
_class
loc:@hidden-bias*
use_locking( *
T0*
_output_shapes
:(
R
Sub_1Subx-input	Sigmoid_3*
T0*'
_output_shapes
:џџџџџџџџџ!
Z
Mean_1/reduction_indicesConst*
dtype0*
value	B : *
_output_shapes
: 
q
Mean_1MeanSub_1Mean_1/reduction_indices*
_output_shapes
:!*
T0*
	keep_dims( *

Tidx0
L
Mul_1/xConst*
dtype0*
valueB
 *
з#<*
_output_shapes
: 
B
Mul_1MulMul_1/xMean_1*
T0*
_output_shapes
:!

AssignAdd_2	AssignAddvisible-biasMul_1*
_class
loc:@visible-bias*
use_locking( *
T0*
_output_shapes
:!
U
cost/SubSubx-input	Sigmoid_3*
T0*'
_output_shapes
:џџџџџџџџџ!
Q
cost/SquareSquarecost/Sub*
T0*'
_output_shapes
:џџџџџџџџџ!
[

cost/ConstConst*
dtype0*
valueB"       *
_output_shapes
:
h
	cost/MeanMeancost/Square
cost/Const*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
=
	cost/SqrtSqrt	cost/Mean*
T0*
_output_shapes
: 
_
ScalarSummary/tagsConst*
dtype0*
valueB Bmean_squared*
_output_shapes
: 
^
ScalarSummaryScalarSummaryScalarSummary/tags	cost/Sqrt*
T0*
_output_shapes
: 
Y
MergeSummary/MergeSummaryMergeSummaryScalarSummary*
_output_shapes
: *
N
H
initNoOp^weights/Assign^hidden-bias/Assign^visible-bias/Assign
P

save/ConstConst*
dtype0*
valueB Bmodel*
_output_shapes
: 

save/SaveV2/tensor_namesConst*
dtype0*7
value.B,Bhidden-biasBvisible-biasBweights*
_output_shapes
:
i
save/SaveV2/shape_and_slicesConst*
dtype0*
valueBB B B *
_output_shapes
:

save/SaveV2SaveV2
save/Constsave/SaveV2/tensor_namessave/SaveV2/shape_and_sliceshidden-biasvisible-biasweights*
dtypes
2
}
save/control_dependencyIdentity
save/Const^save/SaveV2*
_class
loc:@save/Const*
T0*
_output_shapes
: 
o
save/RestoreV2/tensor_namesConst*
dtype0* 
valueBBhidden-bias*
_output_shapes
:
h
save/RestoreV2/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices*
dtypes
2*
_output_shapes
:
 
save/AssignAssignhidden-biassave/RestoreV2*
validate_shape(*
_class
loc:@hidden-bias*
use_locking(*
T0*
_output_shapes
:(
r
save/RestoreV2_1/tensor_namesConst*
dtype0*!
valueBBvisible-bias*
_output_shapes
:
j
!save/RestoreV2_1/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_1	RestoreV2
save/Constsave/RestoreV2_1/tensor_names!save/RestoreV2_1/shape_and_slices*
dtypes
2*
_output_shapes
:
І
save/Assign_1Assignvisible-biassave/RestoreV2_1*
validate_shape(*
_class
loc:@visible-bias*
use_locking(*
T0*
_output_shapes
:!
m
save/RestoreV2_2/tensor_namesConst*
dtype0*
valueBBweights*
_output_shapes
:
j
!save/RestoreV2_2/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_2	RestoreV2
save/Constsave/RestoreV2_2/tensor_names!save/RestoreV2_2/shape_and_slices*
dtypes
2*
_output_shapes
:
 
save/Assign_2Assignweightssave/RestoreV2_2*
validate_shape(*
_class
loc:@weights*
use_locking(*
T0*
_output_shapes

:!(
F
save/restore_allNoOp^save/Assign^save/Assign_1^save/Assign_2"­G	;      цR	hAјћ
жAJ§u
ЦУ
9
Add
x"T
y"T
z"T"
Ttype:
2	
x
Assign
ref"T

value"T

output_ref"T"	
Ttype"
validate_shapebool("
use_lockingbool(
p
	AssignAdd
ref"T

value"T

output_ref"T"
Ttype:
2	"
use_lockingbool( 
8
Const
output"dtype"
valuetensor"
dtypetype
9
Div
x"T
y"T
z"T"
Ttype:
2	
.
Identity

input"T
output"T"	
Ttype
o
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2

Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( "
Ttype:
2	"
Tidxtype0:
2	
8
MergeSummary
inputs*N
summary"
Nint(0
<
Mul
x"T
y"T
z"T"
Ttype:
2	

NoOp
A
Placeholder
output"dtype"
dtypetype"
shapeshape: 
`
Range
start"Tidx
limit"Tidx
delta"Tidx
output"Tidx"
Tidxtype0:
2	
)
Rank

input"T

output"	
Ttype
A
Relu
features"T
activations"T"
Ttype:
2		
l
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
i
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
M
ScalarSummary
tags
values"T
summary"
Ttype:
2		
/
Sigmoid
x"T
y"T"
Ttype:	
2
.
Sign
x"T
y"T"
Ttype:
	2	
,
Sqrt
x"T
y"T"
Ttype:	
2
0
Square
x"T
y"T"
Ttype:
	2	
5
Sub
x"T
y"T
z"T"
Ttype:
	2	
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	

TruncatedNormal

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	
q
Variable
ref"dtype"
shapeshape"
dtypetype"
	containerstring "
shared_namestring c
Y
x-inputPlaceholder*
dtype0*
shape: *'
_output_shapes
:џџџџџџџџџ!
W
hrandPlaceholder*
dtype0*
shape: *'
_output_shapes
:џџџџџџџџџ(
W
vrandPlaceholder*
dtype0*
shape: *'
_output_shapes
:џџџџџџџџџ!
N
PlaceholderPlaceholder*
dtype0*
shape: *
_output_shapes
:
M

keep-probsPlaceholder*
dtype0*
shape: *
_output_shapes
:
g
truncated_normal/shapeConst*
dtype0*
valueB"!   (   *
_output_shapes
:
Z
truncated_normal/meanConst*
dtype0*
valueB
 *    *
_output_shapes
: 
\
truncated_normal/stddevConst*
dtype0*
valueB
 *ЭЬЬ=*
_output_shapes
: 

 truncated_normal/TruncatedNormalTruncatedNormaltruncated_normal/shape*
dtype0*
seed2 *

seed *
T0*
_output_shapes

:!(

truncated_normal/mulMul truncated_normal/TruncatedNormaltruncated_normal/stddev*
T0*
_output_shapes

:!(
m
truncated_normalAddtruncated_normal/multruncated_normal/mean*
T0*
_output_shapes

:!(
y
weightsVariable*
dtype0*
shape
:!(*
shared_name *
	container *
_output_shapes

:!(
Ё
weights/AssignAssignweightstruncated_normal*
validate_shape(*
_class
loc:@weights*
use_locking(*
T0*
_output_shapes

:!(
f
weights/readIdentityweights*
_class
loc:@weights*
T0*
_output_shapes

:!(
R
ConstConst*
dtype0*
valueB(*ЭЬЬ=*
_output_shapes
:(
u
hidden-biasVariable*
dtype0*
shape:(*
shared_name *
	container *
_output_shapes
:(

hidden-bias/AssignAssignhidden-biasConst*
validate_shape(*
_class
loc:@hidden-bias*
use_locking(*
T0*
_output_shapes
:(
n
hidden-bias/readIdentityhidden-bias*
_class
loc:@hidden-bias*
T0*
_output_shapes
:(
T
Const_1Const*
dtype0*
valueB!*ЭЬЬ=*
_output_shapes
:!
v
visible-biasVariable*
dtype0*
shape:!*
shared_name *
	container *
_output_shapes
:!
Ѓ
visible-bias/AssignAssignvisible-biasConst_1*
validate_shape(*
_class
loc:@visible-bias*
use_locking(*
T0*
_output_shapes
:!
q
visible-bias/readIdentityvisible-bias*
_class
loc:@visible-bias*
T0*
_output_shapes
:!

MatMulMatMulx-inputweights/read*
transpose_b( *
transpose_a( *
T0*'
_output_shapes
:џџџџџџџџџ(
V
AddAddMatMulhidden-bias/read*
T0*'
_output_shapes
:џџџџџџџџџ(
I
SigmoidSigmoidAdd*
T0*'
_output_shapes
:џџџџџџџџџ(
L
subSubSigmoidhrand*
T0*'
_output_shapes
:џџџџџџџџџ(
C
SignSignsub*
T0*'
_output_shapes
:џџџџџџџџџ(
D
ReluReluSign*
T0*'
_output_shapes
:џџџџџџџџџ(
E
transpose/RankRankweights/read*
T0*
_output_shapes
: 
Q
transpose/sub/yConst*
dtype0*
value	B :*
_output_shapes
: 
V
transpose/subSubtranspose/Ranktranspose/sub/y*
T0*
_output_shapes
: 
W
transpose/Range/startConst*
dtype0*
value	B : *
_output_shapes
: 
W
transpose/Range/deltaConst*
dtype0*
value	B :*
_output_shapes
: 
~
transpose/RangeRangetranspose/Range/starttranspose/Ranktranspose/Range/delta*

Tidx0*
_output_shapes
:
[
transpose/sub_1Subtranspose/subtranspose/Range*
T0*
_output_shapes
:
k
	transpose	Transposeweights/readtranspose/sub_1*
Tperm0*
T0*
_output_shapes

:(!
~
MatMul_1MatMulSigmoid	transpose*
transpose_b( *
transpose_a( *
T0*'
_output_shapes
:џџџџџџџџџ!
[
Add_1AddMatMul_1visible-bias/read*
T0*'
_output_shapes
:џџџџџџџџџ!
M
	Sigmoid_1SigmoidAdd_1*
T0*'
_output_shapes
:џџџџџџџџџ!

MatMul_2MatMulx-inputweights/read*
transpose_b( *
transpose_a( *
T0*'
_output_shapes
:џџџџџџџџџ(
Z
Add_2AddMatMul_2hidden-bias/read*
T0*'
_output_shapes
:џџџџџџџџџ(
M
	Sigmoid_2SigmoidAdd_2*
T0*'
_output_shapes
:џџџџџџџџџ(
P
sub_1Sub	Sigmoid_2hrand*
T0*'
_output_shapes
:џџџџџџџџџ(
G
Sign_1Signsub_1*
T0*'
_output_shapes
:џџџџџџџџџ(
H
Relu_1ReluSign_1*
T0*'
_output_shapes
:џџџџџџџџџ(
G
transpose_1/RankRankweights/read*
T0*
_output_shapes
: 
S
transpose_1/sub/yConst*
dtype0*
value	B :*
_output_shapes
: 
\
transpose_1/subSubtranspose_1/Ranktranspose_1/sub/y*
T0*
_output_shapes
: 
Y
transpose_1/Range/startConst*
dtype0*
value	B : *
_output_shapes
: 
Y
transpose_1/Range/deltaConst*
dtype0*
value	B :*
_output_shapes
: 

transpose_1/RangeRangetranspose_1/Range/starttranspose_1/Ranktranspose_1/Range/delta*

Tidx0*
_output_shapes
:
a
transpose_1/sub_1Subtranspose_1/subtranspose_1/Range*
T0*
_output_shapes
:
o
transpose_1	Transposeweights/readtranspose_1/sub_1*
Tperm0*
T0*
_output_shapes

:(!

MatMul_3MatMul	Sigmoid_2transpose_1*
transpose_b( *
transpose_a( *
T0*'
_output_shapes
:џџџџџџџџџ!
[
Add_3AddMatMul_3visible-bias/read*
T0*'
_output_shapes
:џџџџџџџџџ!
M
	Sigmoid_3SigmoidAdd_3*
T0*'
_output_shapes
:џџџџџџџџџ!

MatMul_4MatMul	Sigmoid_3weights/read*
transpose_b( *
transpose_a( *
T0*'
_output_shapes
:џџџџџџџџџ(
Z
Add_4AddMatMul_4hidden-bias/read*
T0*'
_output_shapes
:џџџџџџџџџ(
M
	Sigmoid_4SigmoidAdd_4*
T0*'
_output_shapes
:џџџџџџџџџ(
P
sub_2Sub	Sigmoid_4hrand*
T0*'
_output_shapes
:џџџџџџџџџ(
G
Sign_2Signsub_2*
T0*'
_output_shapes
:џџџџџџџџџ(
H
Relu_2ReluSign_2*
T0*'
_output_shapes
:џџџџџџџџџ(
B
transpose_2/RankRankx-input*
T0*
_output_shapes
: 
S
transpose_2/sub/yConst*
dtype0*
value	B :*
_output_shapes
: 
\
transpose_2/subSubtranspose_2/Ranktranspose_2/sub/y*
T0*
_output_shapes
: 
Y
transpose_2/Range/startConst*
dtype0*
value	B : *
_output_shapes
: 
Y
transpose_2/Range/deltaConst*
dtype0*
value	B :*
_output_shapes
: 

transpose_2/RangeRangetranspose_2/Range/starttranspose_2/Ranktranspose_2/Range/delta*

Tidx0*
_output_shapes
:
a
transpose_2/sub_1Subtranspose_2/subtranspose_2/Range*
T0*
_output_shapes
:
s
transpose_2	Transposex-inputtranspose_2/sub_1*
Tperm0*
T0*'
_output_shapes
:!џџџџџџџџџ
v
MatMul_5MatMultranspose_2Relu_1*
transpose_b( *
transpose_a( *
T0*
_output_shapes

:!(
D
transpose_3/RankRank	Sigmoid_3*
T0*
_output_shapes
: 
S
transpose_3/sub/yConst*
dtype0*
value	B :*
_output_shapes
: 
\
transpose_3/subSubtranspose_3/Ranktranspose_3/sub/y*
T0*
_output_shapes
: 
Y
transpose_3/Range/startConst*
dtype0*
value	B : *
_output_shapes
: 
Y
transpose_3/Range/deltaConst*
dtype0*
value	B :*
_output_shapes
: 

transpose_3/RangeRangetranspose_3/Range/starttranspose_3/Ranktranspose_3/Range/delta*

Tidx0*
_output_shapes
:
a
transpose_3/sub_1Subtranspose_3/subtranspose_3/Range*
T0*
_output_shapes
:
u
transpose_3	Transpose	Sigmoid_3transpose_3/sub_1*
Tperm0*
T0*'
_output_shapes
:!џџџџџџџџџ
y
MatMul_6MatMultranspose_3	Sigmoid_4*
transpose_b( *
transpose_a( *
T0*
_output_shapes

:!(
I
sub_3SubMatMul_5MatMul_6*
T0*
_output_shapes

:!(
J
mul/xConst*
dtype0*
valueB
 *
з#<*
_output_shapes
: 
A
mulMulmul/xsub_3*
T0*
_output_shapes

:!(
N
	truediv/yConst*
dtype0*
valueB
 *   A*
_output_shapes
: 
G
truedivDivmul	truediv/y*
T0*
_output_shapes

:!(

	AssignAdd	AssignAddweightstruediv*
_class
loc:@weights*
use_locking( *
T0*
_output_shapes

:!(
R
SubSub	Sigmoid_2	Sigmoid_4*
T0*'
_output_shapes
:џџџџџџџџџ(
X
Mean/reduction_indicesConst*
dtype0*
value	B : *
_output_shapes
: 
k
MeanMeanSubMean/reduction_indices*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:(
J
Mul/xConst*
dtype0*
valueB
 *
з#<*
_output_shapes
: 
<
MulMulMul/xMean*
T0*
_output_shapes
:(

AssignAdd_1	AssignAddhidden-biasMul*
_class
loc:@hidden-bias*
use_locking( *
T0*
_output_shapes
:(
R
Sub_1Subx-input	Sigmoid_3*
T0*'
_output_shapes
:џџџџџџџџџ!
Z
Mean_1/reduction_indicesConst*
dtype0*
value	B : *
_output_shapes
: 
q
Mean_1MeanSub_1Mean_1/reduction_indices*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:!
L
Mul_1/xConst*
dtype0*
valueB
 *
з#<*
_output_shapes
: 
B
Mul_1MulMul_1/xMean_1*
T0*
_output_shapes
:!

AssignAdd_2	AssignAddvisible-biasMul_1*
_class
loc:@visible-bias*
use_locking( *
T0*
_output_shapes
:!
U
cost/SubSubx-input	Sigmoid_3*
T0*'
_output_shapes
:џџџџџџџџџ!
Q
cost/SquareSquarecost/Sub*
T0*'
_output_shapes
:џџџџџџџџџ!
[

cost/ConstConst*
dtype0*
valueB"       *
_output_shapes
:
h
	cost/MeanMeancost/Square
cost/Const*

Tidx0*
T0*
	keep_dims( *
_output_shapes
: 
=
	cost/SqrtSqrt	cost/Mean*
T0*
_output_shapes
: 
_
ScalarSummary/tagsConst*
dtype0*
valueB Bmean_squared*
_output_shapes
: 
^
ScalarSummaryScalarSummaryScalarSummary/tags	cost/Sqrt*
T0*
_output_shapes
: 
Y
MergeSummary/MergeSummaryMergeSummaryScalarSummary*
N*
_output_shapes
: 
H
initNoOp^weights/Assign^hidden-bias/Assign^visible-bias/Assign
P

save/ConstConst*
dtype0*
valueB Bmodel*
_output_shapes
: 

save/SaveV2/tensor_namesConst*
dtype0*7
value.B,Bhidden-biasBvisible-biasBweights*
_output_shapes
:
i
save/SaveV2/shape_and_slicesConst*
dtype0*
valueBB B B *
_output_shapes
:

save/SaveV2SaveV2
save/Constsave/SaveV2/tensor_namessave/SaveV2/shape_and_sliceshidden-biasvisible-biasweights*
dtypes
2
}
save/control_dependencyIdentity
save/Const^save/SaveV2*
_class
loc:@save/Const*
T0*
_output_shapes
: 
o
save/RestoreV2/tensor_namesConst*
dtype0* 
valueBBhidden-bias*
_output_shapes
:
h
save/RestoreV2/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices*
dtypes
2*
_output_shapes
:
 
save/AssignAssignhidden-biassave/RestoreV2*
validate_shape(*
_class
loc:@hidden-bias*
use_locking(*
T0*
_output_shapes
:(
r
save/RestoreV2_1/tensor_namesConst*
dtype0*!
valueBBvisible-bias*
_output_shapes
:
j
!save/RestoreV2_1/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_1	RestoreV2
save/Constsave/RestoreV2_1/tensor_names!save/RestoreV2_1/shape_and_slices*
dtypes
2*
_output_shapes
:
І
save/Assign_1Assignvisible-biassave/RestoreV2_1*
validate_shape(*
_class
loc:@visible-bias*
use_locking(*
T0*
_output_shapes
:!
m
save/RestoreV2_2/tensor_namesConst*
dtype0*
valueBBweights*
_output_shapes
:
j
!save/RestoreV2_2/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_2	RestoreV2
save/Constsave/RestoreV2_2/tensor_names!save/RestoreV2_2/shape_and_slices*
dtypes
2*
_output_shapes
:
 
save/Assign_2Assignweightssave/RestoreV2_2*
validate_shape(*
_class
loc:@weights*
use_locking(*
T0*
_output_shapes

:!(
F
save/restore_allNoOp^save/Assign^save/Assign_1^save/Assign_2""Г
	variablesЅЂ
+
	weights:0weights/Assignweights/read:0
7
hidden-bias:0hidden-bias/Assignhidden-bias/read:0
:
visible-bias:0visible-bias/Assignvisible-bias/read:0" 
	summaries

ScalarSummary:0"Н
trainable_variablesЅЂ
+
	weights:0weights/Assignweights/read:0
7
hidden-bias:0hidden-bias/Assignhidden-bias/read:0
:
visible-bias:0visible-bias/Assignvisible-bias/read:0шoZ        )эЉP	є
жA*

mean_squaredп<=?!R"       x=§	<x;
жA*

mean_squared"=5/Лд"       x=§	~+Z
жA*

mean_squaredђZ=Nэ "       x=§	/z
жA*

mean_squared sэ<Ягџ"       x=§	НШ,
жA*

mean_squaredFл<ЇЕоЌ"       x=§	лЃЙ
жA*

mean_squared3шв<№pх"       x=§	ФAюи
жA*

mean_squared`pЮ<p7иж"       x=§	  ј
жA*

mean_squaredЏУ<бод4"       x=§	Кџ
жA*

mean_squaredЖЉЛ<gMNе"       x=§	h^8
жA	*

mean_squaredXЛЕ<УЄЭ