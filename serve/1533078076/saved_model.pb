ÎŁ
1ç0
:
Add
x"T
y"T
z"T"
Ttype:
2	
h
All	
input

reduction_indices"Tidx

output
"
	keep_dimsbool( "
Tidxtype0:
2	

ArgMax

input"T
	dimension"Tidx
output"output_type" 
Ttype:
2	"
Tidxtype0:
2	"
output_typetype0	:
2	
P
Assert
	condition
	
data2T"
T
list(type)(0"
	summarizeint
x
Assign
ref"T

value"T

output_ref"T"	
Ttype"
validate_shapebool("
use_lockingbool(
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
ě
Conv2D

input"T
filter"T
output"T"
Ttype:
2"
strides	list(int)"
use_cudnn_on_gpubool(""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

y
Enter	
data"T
output"T"	
Ttype"

frame_namestring"
is_constantbool( "
parallel_iterationsint

B
Equal
x"T
y"T
z
"
Ttype:
2	

)
Exit	
data"T
output"T"	
Ttype
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
B
GreaterEqual
x"T
y"T
z
"
Ttype:
2	
.
Identity

input"T
output"T"	
Ttype
:
Less
x"T
y"T
z
"
Ttype:
2	
$

LogicalAnd
x

y

z

!
LoopCond	
input


output

p
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
	2

Max

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
;
Maximum
x"T
y"T
z"T"
Ttype:

2	
N
Merge
inputs"T*N
output"T
value_index"	
Ttype"
Nint(0
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(

Min

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
;
Minimum
x"T
y"T
z"T"
Ttype:

2	
=
Mul
x"T
y"T
z"T"
Ttype:
2	
2
NextIteration	
data"T
output"T"	
Ttype

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
ď
ParseExample

serialized	
names
sparse_keys*Nsparse

dense_keys*Ndense
dense_defaults2Tdense
sparse_indices	*Nsparse
sparse_values2sparse_types
sparse_shapes	*Nsparse
dense_values2Tdense"
Nsparseint("
Ndenseint("%
sparse_types
list(type)(:
2	"
Tdense
list(type)(:
2	"
dense_shapeslist(shape)(
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
~
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	
a
Range
start"Tidx
limit"Tidx
delta"Tidx
output"Tidx"
Tidxtype0:	
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0

ReverseSequence

input"T
seq_lengths"Tlen
output"T"
seq_dimint"
	batch_dimint "	
Ttype"
Tlentype0	:
2	
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
a
Slice

input"T
begin"Index
size"Index
output"T"	
Ttype"
Indextype:
2	
ź
SparseToDense
sparse_indices"Tindices
output_shape"Tindices
sparse_values"T
default_value"T

dense"T"
validate_indicesbool("	
Ttype"
Tindicestype:
2	
[
Split
	split_dim

value"T
output"T*	num_split"
	num_splitint(0"	
Ttype
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
ö
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
:
Sub
x"T
y"T
z"T"
Ttype:
2	

Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
M
Switch	
data"T
pred

output_false"T
output_true"T"	
Ttype
-
Tanh
x"T
y"T"
Ttype:

2
{
TensorArrayGatherV3

handle
indices
flow_in
value"dtype"
dtypetype"
element_shapeshape:
Y
TensorArrayReadV3

handle	
index
flow_in
value"dtype"
dtypetype
d
TensorArrayScatterV3

handle
indices

value"T
flow_in
flow_out"	
Ttype
9
TensorArraySizeV3

handle
flow_in
size
Ţ
TensorArrayV3
size

handle
flow"
dtypetype"
element_shapeshape:"
dynamic_sizebool( "
clear_after_readbool("$
identical_element_shapesbool( "
tensor_array_namestring 
`
TensorArrayWriteV3

handle	
index

value"T
flow_in
flow_out"	
Ttype
c
Tile

input"T
	multiples"
Tmultiples
output"T"	
Ttype"

Tmultiplestype0:
2	
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
s

VariableV2
ref"dtype"
shapeshape"
dtypetype"
	containerstring "
shared_namestring 
&
	ZerosLike
x"T
y"T"	
Ttype"serve*1.8.02v1.8.0-0-g93bc2e20728šň

global_step/Initializer/zerosConst*
value	B	 R *
_class
loc:@global_step*
dtype0	*
_output_shapes
: 
k
global_step
VariableV2*
shape: *
dtype0	*
_output_shapes
: *
_class
loc:@global_step

global_step/AssignAssignglobal_stepglobal_step/Initializer/zeros*
_output_shapes
: *
T0	*
_class
loc:@global_step
j
global_step/readIdentityglobal_step*
T0	*
_class
loc:@global_step*
_output_shapes
: 
o
input_example_tensorPlaceholder*
shape:˙˙˙˙˙˙˙˙˙*
dtype0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
U
ParseExample/ConstConst*
dtype0	*
_output_shapes
: *
valueB	 
W
ParseExample/Const_1Const*
valueB	 *
dtype0	*
_output_shapes
: 
b
ParseExample/ParseExample/namesConst*
dtype0*
_output_shapes
: *
valueB 
k
'ParseExample/ParseExample/sparse_keys_0Const*
dtype0*
_output_shapes
: *
valueB	 Bink
r
&ParseExample/ParseExample/dense_keys_0Const*
valueB Bclass_index*
dtype0*
_output_shapes
: 
l
&ParseExample/ParseExample/dense_keys_1Const*
dtype0*
_output_shapes
: *
valueB Bshape
Í
ParseExample/ParseExampleParseExampleinput_example_tensorParseExample/ParseExample/names'ParseExample/ParseExample/sparse_keys_0&ParseExample/ParseExample/dense_keys_0&ParseExample/ParseExample/dense_keys_1ParseExample/ConstParseExample/Const_1*
Tdense
2		*
Ndense*b
_output_shapesP
N:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙::˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
Nsparse*
dense_shapes
::*
sparse_types
2
`
SparseToDense/default_valueConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Ű
SparseToDenseSparseToDenseParseExample/ParseExampleParseExample/ParseExample:2ParseExample/ParseExample:1SparseToDense/default_value*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
Tindices0	
\
Slice/beginConst*
dtype0*
_output_shapes
:*
valueB"        
[

Slice/sizeConst*
dtype0*
_output_shapes
:*
valueB"      
z
SliceSliceParseExample/ParseExample:4Slice/begin
Slice/size*
T0	*
Index0*
_output_shapes

:
>
SqueezeSqueezeSlice*
T0	*
_output_shapes
:
b
Reshape/shapeConst*!
valueB"   ˙˙˙˙   *
dtype0*
_output_shapes
:
f
ReshapeReshapeSparseToDenseReshape/shape*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙
Š
0conv1d_0/kernel/Initializer/random_uniform/shapeConst*!
valueB"      0   *"
_class
loc:@conv1d_0/kernel*
dtype0*
_output_shapes
:

.conv1d_0/kernel/Initializer/random_uniform/minConst*
valueB
 *ž*"
_class
loc:@conv1d_0/kernel*
dtype0*
_output_shapes
: 

.conv1d_0/kernel/Initializer/random_uniform/maxConst*
valueB
 *>*"
_class
loc:@conv1d_0/kernel*
dtype0*
_output_shapes
: 
Ů
8conv1d_0/kernel/Initializer/random_uniform/RandomUniformRandomUniform0conv1d_0/kernel/Initializer/random_uniform/shape*
dtype0*"
_output_shapes
:0*
T0*"
_class
loc:@conv1d_0/kernel
Ú
.conv1d_0/kernel/Initializer/random_uniform/subSub.conv1d_0/kernel/Initializer/random_uniform/max.conv1d_0/kernel/Initializer/random_uniform/min*
T0*"
_class
loc:@conv1d_0/kernel*
_output_shapes
: 
đ
.conv1d_0/kernel/Initializer/random_uniform/mulMul8conv1d_0/kernel/Initializer/random_uniform/RandomUniform.conv1d_0/kernel/Initializer/random_uniform/sub*"
_output_shapes
:0*
T0*"
_class
loc:@conv1d_0/kernel
â
*conv1d_0/kernel/Initializer/random_uniformAdd.conv1d_0/kernel/Initializer/random_uniform/mul.conv1d_0/kernel/Initializer/random_uniform/min*
T0*"
_class
loc:@conv1d_0/kernel*"
_output_shapes
:0

conv1d_0/kernel
VariableV2*
dtype0*"
_output_shapes
:0*"
_class
loc:@conv1d_0/kernel*
shape:0
Ž
conv1d_0/kernel/AssignAssignconv1d_0/kernel*conv1d_0/kernel/Initializer/random_uniform*
T0*"
_class
loc:@conv1d_0/kernel*"
_output_shapes
:0

conv1d_0/kernel/readIdentityconv1d_0/kernel*"
_output_shapes
:0*
T0*"
_class
loc:@conv1d_0/kernel

conv1d_0/bias/Initializer/zerosConst*
dtype0*
_output_shapes
:0*
valueB0*    * 
_class
loc:@conv1d_0/bias
w
conv1d_0/bias
VariableV2*
shape:0*
dtype0*
_output_shapes
:0* 
_class
loc:@conv1d_0/bias

conv1d_0/bias/AssignAssignconv1d_0/biasconv1d_0/bias/Initializer/zeros*
_output_shapes
:0*
T0* 
_class
loc:@conv1d_0/bias
t
conv1d_0/bias/readIdentityconv1d_0/bias*
T0* 
_class
loc:@conv1d_0/bias*
_output_shapes
:0
`
conv1d_0/dilation_rateConst*
dtype0*
_output_shapes
:*
valueB:
`
conv1d_0/conv1d/ExpandDims/dimConst*
value	B :*
dtype0*
_output_shapes
: 

conv1d_0/conv1d/ExpandDims
ExpandDimsReshapeconv1d_0/conv1d/ExpandDims/dim*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙
b
 conv1d_0/conv1d/ExpandDims_1/dimConst*
value	B : *
dtype0*
_output_shapes
: 

conv1d_0/conv1d/ExpandDims_1
ExpandDimsconv1d_0/kernel/read conv1d_0/conv1d/ExpandDims_1/dim*&
_output_shapes
:0*
T0
ź
conv1d_0/conv1d/Conv2DConv2Dconv1d_0/conv1d/ExpandDimsconv1d_0/conv1d/ExpandDims_1*
strides
*
paddingSAME*/
_output_shapes
:˙˙˙˙˙˙˙˙˙0*
T0

conv1d_0/conv1d/SqueezeSqueezeconv1d_0/conv1d/Conv2D*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙0*
squeeze_dims

~
conv1d_0/BiasAddBiasAddconv1d_0/conv1d/Squeezeconv1d_0/bias/read*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙0
d
dropout/IdentityIdentityconv1d_0/BiasAdd*+
_output_shapes
:˙˙˙˙˙˙˙˙˙0*
T0
Š
0conv1d_1/kernel/Initializer/random_uniform/shapeConst*!
valueB"   0   @   *"
_class
loc:@conv1d_1/kernel*
dtype0*
_output_shapes
:

.conv1d_1/kernel/Initializer/random_uniform/minConst*
valueB
 *÷üÓ˝*"
_class
loc:@conv1d_1/kernel*
dtype0*
_output_shapes
: 

.conv1d_1/kernel/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *
valueB
 *÷üÓ=*"
_class
loc:@conv1d_1/kernel
Ů
8conv1d_1/kernel/Initializer/random_uniform/RandomUniformRandomUniform0conv1d_1/kernel/Initializer/random_uniform/shape*
dtype0*"
_output_shapes
:0@*
T0*"
_class
loc:@conv1d_1/kernel
Ú
.conv1d_1/kernel/Initializer/random_uniform/subSub.conv1d_1/kernel/Initializer/random_uniform/max.conv1d_1/kernel/Initializer/random_uniform/min*
T0*"
_class
loc:@conv1d_1/kernel*
_output_shapes
: 
đ
.conv1d_1/kernel/Initializer/random_uniform/mulMul8conv1d_1/kernel/Initializer/random_uniform/RandomUniform.conv1d_1/kernel/Initializer/random_uniform/sub*
T0*"
_class
loc:@conv1d_1/kernel*"
_output_shapes
:0@
â
*conv1d_1/kernel/Initializer/random_uniformAdd.conv1d_1/kernel/Initializer/random_uniform/mul.conv1d_1/kernel/Initializer/random_uniform/min*"
_output_shapes
:0@*
T0*"
_class
loc:@conv1d_1/kernel

conv1d_1/kernel
VariableV2*
shape:0@*
dtype0*"
_output_shapes
:0@*"
_class
loc:@conv1d_1/kernel
Ž
conv1d_1/kernel/AssignAssignconv1d_1/kernel*conv1d_1/kernel/Initializer/random_uniform*"
_output_shapes
:0@*
T0*"
_class
loc:@conv1d_1/kernel

conv1d_1/kernel/readIdentityconv1d_1/kernel*
T0*"
_class
loc:@conv1d_1/kernel*"
_output_shapes
:0@

conv1d_1/bias/Initializer/zerosConst*
valueB@*    * 
_class
loc:@conv1d_1/bias*
dtype0*
_output_shapes
:@
w
conv1d_1/bias
VariableV2* 
_class
loc:@conv1d_1/bias*
shape:@*
dtype0*
_output_shapes
:@

conv1d_1/bias/AssignAssignconv1d_1/biasconv1d_1/bias/Initializer/zeros*
T0* 
_class
loc:@conv1d_1/bias*
_output_shapes
:@
t
conv1d_1/bias/readIdentityconv1d_1/bias*
T0* 
_class
loc:@conv1d_1/bias*
_output_shapes
:@
`
conv1d_1/dilation_rateConst*
valueB:*
dtype0*
_output_shapes
:
`
conv1d_1/conv1d/ExpandDims/dimConst*
value	B :*
dtype0*
_output_shapes
: 

conv1d_1/conv1d/ExpandDims
ExpandDimsdropout/Identityconv1d_1/conv1d/ExpandDims/dim*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙0
b
 conv1d_1/conv1d/ExpandDims_1/dimConst*
dtype0*
_output_shapes
: *
value	B : 

conv1d_1/conv1d/ExpandDims_1
ExpandDimsconv1d_1/kernel/read conv1d_1/conv1d/ExpandDims_1/dim*
T0*&
_output_shapes
:0@
ź
conv1d_1/conv1d/Conv2DConv2Dconv1d_1/conv1d/ExpandDimsconv1d_1/conv1d/ExpandDims_1*
strides
*
paddingSAME*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@*
T0

conv1d_1/conv1d/SqueezeSqueezeconv1d_1/conv1d/Conv2D*
squeeze_dims
*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙@
~
conv1d_1/BiasAddBiasAddconv1d_1/conv1d/Squeezeconv1d_1/bias/read*+
_output_shapes
:˙˙˙˙˙˙˙˙˙@*
T0
f
dropout_1/IdentityIdentityconv1d_1/BiasAdd*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙@
Š
0conv1d_2/kernel/Initializer/random_uniform/shapeConst*!
valueB"   @   `   *"
_class
loc:@conv1d_2/kernel*
dtype0*
_output_shapes
:

.conv1d_2/kernel/Initializer/random_uniform/minConst*
valueB
 *.ůä˝*"
_class
loc:@conv1d_2/kernel*
dtype0*
_output_shapes
: 

.conv1d_2/kernel/Initializer/random_uniform/maxConst*
valueB
 *.ůä=*"
_class
loc:@conv1d_2/kernel*
dtype0*
_output_shapes
: 
Ů
8conv1d_2/kernel/Initializer/random_uniform/RandomUniformRandomUniform0conv1d_2/kernel/Initializer/random_uniform/shape*
dtype0*"
_output_shapes
:@`*
T0*"
_class
loc:@conv1d_2/kernel
Ú
.conv1d_2/kernel/Initializer/random_uniform/subSub.conv1d_2/kernel/Initializer/random_uniform/max.conv1d_2/kernel/Initializer/random_uniform/min*
T0*"
_class
loc:@conv1d_2/kernel*
_output_shapes
: 
đ
.conv1d_2/kernel/Initializer/random_uniform/mulMul8conv1d_2/kernel/Initializer/random_uniform/RandomUniform.conv1d_2/kernel/Initializer/random_uniform/sub*
T0*"
_class
loc:@conv1d_2/kernel*"
_output_shapes
:@`
â
*conv1d_2/kernel/Initializer/random_uniformAdd.conv1d_2/kernel/Initializer/random_uniform/mul.conv1d_2/kernel/Initializer/random_uniform/min*
T0*"
_class
loc:@conv1d_2/kernel*"
_output_shapes
:@`

conv1d_2/kernel
VariableV2*
dtype0*"
_output_shapes
:@`*"
_class
loc:@conv1d_2/kernel*
shape:@`
Ž
conv1d_2/kernel/AssignAssignconv1d_2/kernel*conv1d_2/kernel/Initializer/random_uniform*
T0*"
_class
loc:@conv1d_2/kernel*"
_output_shapes
:@`

conv1d_2/kernel/readIdentityconv1d_2/kernel*"
_output_shapes
:@`*
T0*"
_class
loc:@conv1d_2/kernel

conv1d_2/bias/Initializer/zerosConst*
valueB`*    * 
_class
loc:@conv1d_2/bias*
dtype0*
_output_shapes
:`
w
conv1d_2/bias
VariableV2* 
_class
loc:@conv1d_2/bias*
shape:`*
dtype0*
_output_shapes
:`

conv1d_2/bias/AssignAssignconv1d_2/biasconv1d_2/bias/Initializer/zeros*
T0* 
_class
loc:@conv1d_2/bias*
_output_shapes
:`
t
conv1d_2/bias/readIdentityconv1d_2/bias*
T0* 
_class
loc:@conv1d_2/bias*
_output_shapes
:`
`
conv1d_2/dilation_rateConst*
valueB:*
dtype0*
_output_shapes
:
`
conv1d_2/conv1d/ExpandDims/dimConst*
value	B :*
dtype0*
_output_shapes
: 

conv1d_2/conv1d/ExpandDims
ExpandDimsdropout_1/Identityconv1d_2/conv1d/ExpandDims/dim*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@
b
 conv1d_2/conv1d/ExpandDims_1/dimConst*
value	B : *
dtype0*
_output_shapes
: 

conv1d_2/conv1d/ExpandDims_1
ExpandDimsconv1d_2/kernel/read conv1d_2/conv1d/ExpandDims_1/dim*
T0*&
_output_shapes
:@`
ź
conv1d_2/conv1d/Conv2DConv2Dconv1d_2/conv1d/ExpandDimsconv1d_2/conv1d/ExpandDims_1*/
_output_shapes
:˙˙˙˙˙˙˙˙˙`*
T0*
strides
*
paddingSAME

conv1d_2/conv1d/SqueezeSqueezeconv1d_2/conv1d/Conv2D*+
_output_shapes
:˙˙˙˙˙˙˙˙˙`*
squeeze_dims
*
T0
~
conv1d_2/BiasAddBiasAddconv1d_2/conv1d/Squeezeconv1d_2/bias/read*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙`
]
DropoutWrapperInit/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *  ?
_
DropoutWrapperInit/Const_1Const*
valueB
 *  ?*
dtype0*
_output_shapes
: 
_
DropoutWrapperInit/Const_2Const*
valueB
 *  ?*
dtype0*
_output_shapes
: 
_
DropoutWrapperInit_1/ConstConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
a
DropoutWrapperInit_1/Const_1Const*
valueB
 *  ?*
dtype0*
_output_shapes
: 
a
DropoutWrapperInit_1/Const_2Const*
valueB
 *  ?*
dtype0*
_output_shapes
: 
_
DropoutWrapperInit_2/ConstConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
a
DropoutWrapperInit_2/Const_1Const*
dtype0*
_output_shapes
: *
valueB
 *  ?
a
DropoutWrapperInit_2/Const_2Const*
dtype0*
_output_shapes
: *
valueB
 *  ?
_
DropoutWrapperInit_3/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *  ?
a
DropoutWrapperInit_3/Const_1Const*
valueB
 *  ?*
dtype0*
_output_shapes
: 
a
DropoutWrapperInit_3/Const_2Const*
valueB
 *  ?*
dtype0*
_output_shapes
: 
_
DropoutWrapperInit_4/ConstConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
a
DropoutWrapperInit_4/Const_1Const*
dtype0*
_output_shapes
: *
valueB
 *  ?
a
DropoutWrapperInit_4/Const_2Const*
valueB
 *  ?*
dtype0*
_output_shapes
: 
_
DropoutWrapperInit_5/ConstConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
a
DropoutWrapperInit_5/Const_1Const*
valueB
 *  ?*
dtype0*
_output_shapes
: 
a
DropoutWrapperInit_5/Const_2Const*
valueB
 *  ?*
dtype0*
_output_shapes
: 
x
6rnn_classification/cell_0/bidirectional_rnn/fw/fw/RankConst*
value	B :*
dtype0*
_output_shapes
: 

=rnn_classification/cell_0/bidirectional_rnn/fw/fw/range/startConst*
dtype0*
_output_shapes
: *
value	B :

=rnn_classification/cell_0/bidirectional_rnn/fw/fw/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 

7rnn_classification/cell_0/bidirectional_rnn/fw/fw/rangeRange=rnn_classification/cell_0/bidirectional_rnn/fw/fw/range/start6rnn_classification/cell_0/bidirectional_rnn/fw/fw/Rank=rnn_classification/cell_0/bidirectional_rnn/fw/fw/range/delta*
_output_shapes
:

Arnn_classification/cell_0/bidirectional_rnn/fw/fw/concat/values_0Const*
dtype0*
_output_shapes
:*
valueB"       

=rnn_classification/cell_0/bidirectional_rnn/fw/fw/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
­
8rnn_classification/cell_0/bidirectional_rnn/fw/fw/concatConcatV2Arnn_classification/cell_0/bidirectional_rnn/fw/fw/concat/values_07rnn_classification/cell_0/bidirectional_rnn/fw/fw/range=rnn_classification/cell_0/bidirectional_rnn/fw/fw/concat/axis*
N*
_output_shapes
:*
T0
Ę
;rnn_classification/cell_0/bidirectional_rnn/fw/fw/transpose	Transposeconv1d_2/BiasAdd8rnn_classification/cell_0/bidirectional_rnn/fw/fw/concat*+
_output_shapes
:˙˙˙˙˙˙˙˙˙`*
T0
~
9rnn_classification/cell_0/bidirectional_rnn/fw/fw/ToInt32CastSqueeze*

SrcT0	*

DstT0*
_output_shapes
:
­
Arnn_classification/cell_0/bidirectional_rnn/fw/fw/sequence_lengthIdentity9rnn_classification/cell_0/bidirectional_rnn/fw/fw/ToInt32*
T0*
_output_shapes
:
°
frnn_classification/cell_0/bidirectional_rnn/fw/fw/DropoutWrapperZeroState/BasicLSTMCellZeroState/ConstConst*
valueB:*
dtype0*
_output_shapes
:
ł
hrnn_classification/cell_0/bidirectional_rnn/fw/fw/DropoutWrapperZeroState/BasicLSTMCellZeroState/Const_1Const*
valueB:*
dtype0*
_output_shapes
:
Ž
lrnn_classification/cell_0/bidirectional_rnn/fw/fw/DropoutWrapperZeroState/BasicLSTMCellZeroState/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
á
grnn_classification/cell_0/bidirectional_rnn/fw/fw/DropoutWrapperZeroState/BasicLSTMCellZeroState/concatConcatV2frnn_classification/cell_0/bidirectional_rnn/fw/fw/DropoutWrapperZeroState/BasicLSTMCellZeroState/Consthrnn_classification/cell_0/bidirectional_rnn/fw/fw/DropoutWrapperZeroState/BasicLSTMCellZeroState/Const_1lrnn_classification/cell_0/bidirectional_rnn/fw/fw/DropoutWrapperZeroState/BasicLSTMCellZeroState/concat/axis*
T0*
N*
_output_shapes
:
ą
lrnn_classification/cell_0/bidirectional_rnn/fw/fw/DropoutWrapperZeroState/BasicLSTMCellZeroState/zeros/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    
ď
frnn_classification/cell_0/bidirectional_rnn/fw/fw/DropoutWrapperZeroState/BasicLSTMCellZeroState/zerosFillgrnn_classification/cell_0/bidirectional_rnn/fw/fw/DropoutWrapperZeroState/BasicLSTMCellZeroState/concatlrnn_classification/cell_0/bidirectional_rnn/fw/fw/DropoutWrapperZeroState/BasicLSTMCellZeroState/zeros/Const*
T0*
_output_shapes
:	
˛
hrnn_classification/cell_0/bidirectional_rnn/fw/fw/DropoutWrapperZeroState/BasicLSTMCellZeroState/Const_2Const*
valueB:*
dtype0*
_output_shapes
:
ł
hrnn_classification/cell_0/bidirectional_rnn/fw/fw/DropoutWrapperZeroState/BasicLSTMCellZeroState/Const_3Const*
valueB:*
dtype0*
_output_shapes
:
˛
hrnn_classification/cell_0/bidirectional_rnn/fw/fw/DropoutWrapperZeroState/BasicLSTMCellZeroState/Const_4Const*
valueB:*
dtype0*
_output_shapes
:
ł
hrnn_classification/cell_0/bidirectional_rnn/fw/fw/DropoutWrapperZeroState/BasicLSTMCellZeroState/Const_5Const*
valueB:*
dtype0*
_output_shapes
:
°
nrnn_classification/cell_0/bidirectional_rnn/fw/fw/DropoutWrapperZeroState/BasicLSTMCellZeroState/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
ç
irnn_classification/cell_0/bidirectional_rnn/fw/fw/DropoutWrapperZeroState/BasicLSTMCellZeroState/concat_1ConcatV2hrnn_classification/cell_0/bidirectional_rnn/fw/fw/DropoutWrapperZeroState/BasicLSTMCellZeroState/Const_4hrnn_classification/cell_0/bidirectional_rnn/fw/fw/DropoutWrapperZeroState/BasicLSTMCellZeroState/Const_5nrnn_classification/cell_0/bidirectional_rnn/fw/fw/DropoutWrapperZeroState/BasicLSTMCellZeroState/concat_1/axis*
N*
_output_shapes
:*
T0
ł
nrnn_classification/cell_0/bidirectional_rnn/fw/fw/DropoutWrapperZeroState/BasicLSTMCellZeroState/zeros_1/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
ő
hrnn_classification/cell_0/bidirectional_rnn/fw/fw/DropoutWrapperZeroState/BasicLSTMCellZeroState/zeros_1Fillirnn_classification/cell_0/bidirectional_rnn/fw/fw/DropoutWrapperZeroState/BasicLSTMCellZeroState/concat_1nrnn_classification/cell_0/bidirectional_rnn/fw/fw/DropoutWrapperZeroState/BasicLSTMCellZeroState/zeros_1/Const*
_output_shapes
:	*
T0
˛
hrnn_classification/cell_0/bidirectional_rnn/fw/fw/DropoutWrapperZeroState/BasicLSTMCellZeroState/Const_6Const*
valueB:*
dtype0*
_output_shapes
:
ł
hrnn_classification/cell_0/bidirectional_rnn/fw/fw/DropoutWrapperZeroState/BasicLSTMCellZeroState/Const_7Const*
valueB:*
dtype0*
_output_shapes
:

7rnn_classification/cell_0/bidirectional_rnn/fw/fw/ShapeConst*
valueB:*
dtype0*
_output_shapes
:

7rnn_classification/cell_0/bidirectional_rnn/fw/fw/stackConst*
valueB:*
dtype0*
_output_shapes
:
×
7rnn_classification/cell_0/bidirectional_rnn/fw/fw/EqualEqual7rnn_classification/cell_0/bidirectional_rnn/fw/fw/Shape7rnn_classification/cell_0/bidirectional_rnn/fw/fw/stack*
T0*
_output_shapes
:

7rnn_classification/cell_0/bidirectional_rnn/fw/fw/ConstConst*
valueB: *
dtype0*
_output_shapes
:
Ć
5rnn_classification/cell_0/bidirectional_rnn/fw/fw/AllAll7rnn_classification/cell_0/bidirectional_rnn/fw/fw/Equal7rnn_classification/cell_0/bidirectional_rnn/fw/fw/Const*
_output_shapes
: 
ŕ
>rnn_classification/cell_0/bidirectional_rnn/fw/fw/Assert/ConstConst*r
valueiBg BaExpected shape for Tensor rnn_classification/cell_0/bidirectional_rnn/fw/fw/sequence_length:0 is *
dtype0*
_output_shapes
: 

@rnn_classification/cell_0/bidirectional_rnn/fw/fw/Assert/Const_1Const*
dtype0*
_output_shapes
: *!
valueB B but saw shape: 
č
Frnn_classification/cell_0/bidirectional_rnn/fw/fw/Assert/Assert/data_0Const*
dtype0*
_output_shapes
: *r
valueiBg BaExpected shape for Tensor rnn_classification/cell_0/bidirectional_rnn/fw/fw/sequence_length:0 is 

Frnn_classification/cell_0/bidirectional_rnn/fw/fw/Assert/Assert/data_2Const*!
valueB B but saw shape: *
dtype0*
_output_shapes
: 

?rnn_classification/cell_0/bidirectional_rnn/fw/fw/Assert/AssertAssert5rnn_classification/cell_0/bidirectional_rnn/fw/fw/AllFrnn_classification/cell_0/bidirectional_rnn/fw/fw/Assert/Assert/data_07rnn_classification/cell_0/bidirectional_rnn/fw/fw/stackFrnn_classification/cell_0/bidirectional_rnn/fw/fw/Assert/Assert/data_27rnn_classification/cell_0/bidirectional_rnn/fw/fw/Shape*
T
2
ó
=rnn_classification/cell_0/bidirectional_rnn/fw/fw/CheckSeqLenIdentityArnn_classification/cell_0/bidirectional_rnn/fw/fw/sequence_length@^rnn_classification/cell_0/bidirectional_rnn/fw/fw/Assert/Assert*
T0*
_output_shapes
:
¤
9rnn_classification/cell_0/bidirectional_rnn/fw/fw/Shape_1Shape;rnn_classification/cell_0/bidirectional_rnn/fw/fw/transpose*
_output_shapes
:*
T0

Ernn_classification/cell_0/bidirectional_rnn/fw/fw/strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB: 

Grnn_classification/cell_0/bidirectional_rnn/fw/fw/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB:

Grnn_classification/cell_0/bidirectional_rnn/fw/fw/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
Š
?rnn_classification/cell_0/bidirectional_rnn/fw/fw/strided_sliceStridedSlice9rnn_classification/cell_0/bidirectional_rnn/fw/fw/Shape_1Ernn_classification/cell_0/bidirectional_rnn/fw/fw/strided_slice/stackGrnn_classification/cell_0/bidirectional_rnn/fw/fw/strided_slice/stack_1Grnn_classification/cell_0/bidirectional_rnn/fw/fw/strided_slice/stack_2*
Index0*
T0*
shrink_axis_mask*
_output_shapes
: 

9rnn_classification/cell_0/bidirectional_rnn/fw/fw/Const_1Const*
dtype0*
_output_shapes
:*
valueB:

9rnn_classification/cell_0/bidirectional_rnn/fw/fw/Const_2Const*
dtype0*
_output_shapes
:*
valueB:

?rnn_classification/cell_0/bidirectional_rnn/fw/fw/concat_1/axisConst*
dtype0*
_output_shapes
: *
value	B : 
Ť
:rnn_classification/cell_0/bidirectional_rnn/fw/fw/concat_1ConcatV29rnn_classification/cell_0/bidirectional_rnn/fw/fw/Const_19rnn_classification/cell_0/bidirectional_rnn/fw/fw/Const_2?rnn_classification/cell_0/bidirectional_rnn/fw/fw/concat_1/axis*
T0*
N*
_output_shapes
:

=rnn_classification/cell_0/bidirectional_rnn/fw/fw/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
ä
7rnn_classification/cell_0/bidirectional_rnn/fw/fw/zerosFill:rnn_classification/cell_0/bidirectional_rnn/fw/fw/concat_1=rnn_classification/cell_0/bidirectional_rnn/fw/fw/zeros/Const*
T0*
_output_shapes
:	

9rnn_classification/cell_0/bidirectional_rnn/fw/fw/Const_3Const*
valueB: *
dtype0*
_output_shapes
:
×
5rnn_classification/cell_0/bidirectional_rnn/fw/fw/MinMin=rnn_classification/cell_0/bidirectional_rnn/fw/fw/CheckSeqLen9rnn_classification/cell_0/bidirectional_rnn/fw/fw/Const_3*
T0*
_output_shapes
: 

9rnn_classification/cell_0/bidirectional_rnn/fw/fw/Const_4Const*
dtype0*
_output_shapes
:*
valueB: 
×
5rnn_classification/cell_0/bidirectional_rnn/fw/fw/MaxMax=rnn_classification/cell_0/bidirectional_rnn/fw/fw/CheckSeqLen9rnn_classification/cell_0/bidirectional_rnn/fw/fw/Const_4*
T0*
_output_shapes
: 
x
6rnn_classification/cell_0/bidirectional_rnn/fw/fw/timeConst*
value	B : *
dtype0*
_output_shapes
: 
×
=rnn_classification/cell_0/bidirectional_rnn/fw/fw/TensorArrayTensorArrayV3?rnn_classification/cell_0/bidirectional_rnn/fw/fw/strided_slice*
identical_element_shapes(*]
tensor_array_nameHFrnn_classification/cell_0/bidirectional_rnn/fw/fw/dynamic_rnn/output_0*
dtype0*
_output_shapes

:: *
element_shape:	
×
?rnn_classification/cell_0/bidirectional_rnn/fw/fw/TensorArray_1TensorArrayV3?rnn_classification/cell_0/bidirectional_rnn/fw/fw/strided_slice*
dtype0*
_output_shapes

:: *
element_shape
:`*
identical_element_shapes(*\
tensor_array_nameGErnn_classification/cell_0/bidirectional_rnn/fw/fw/dynamic_rnn/input_0
ľ
Jrnn_classification/cell_0/bidirectional_rnn/fw/fw/TensorArrayUnstack/ShapeShape;rnn_classification/cell_0/bidirectional_rnn/fw/fw/transpose*
T0*
_output_shapes
:
˘
Xrnn_classification/cell_0/bidirectional_rnn/fw/fw/TensorArrayUnstack/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
¤
Zrnn_classification/cell_0/bidirectional_rnn/fw/fw/TensorArrayUnstack/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
¤
Zrnn_classification/cell_0/bidirectional_rnn/fw/fw/TensorArrayUnstack/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:

Rrnn_classification/cell_0/bidirectional_rnn/fw/fw/TensorArrayUnstack/strided_sliceStridedSliceJrnn_classification/cell_0/bidirectional_rnn/fw/fw/TensorArrayUnstack/ShapeXrnn_classification/cell_0/bidirectional_rnn/fw/fw/TensorArrayUnstack/strided_slice/stackZrnn_classification/cell_0/bidirectional_rnn/fw/fw/TensorArrayUnstack/strided_slice/stack_1Zrnn_classification/cell_0/bidirectional_rnn/fw/fw/TensorArrayUnstack/strided_slice/stack_2*
T0*
Index0*
shrink_axis_mask*
_output_shapes
: 

Prnn_classification/cell_0/bidirectional_rnn/fw/fw/TensorArrayUnstack/range/startConst*
value	B : *
dtype0*
_output_shapes
: 

Prnn_classification/cell_0/bidirectional_rnn/fw/fw/TensorArrayUnstack/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
đ
Jrnn_classification/cell_0/bidirectional_rnn/fw/fw/TensorArrayUnstack/rangeRangePrnn_classification/cell_0/bidirectional_rnn/fw/fw/TensorArrayUnstack/range/startRrnn_classification/cell_0/bidirectional_rnn/fw/fw/TensorArrayUnstack/strided_slicePrnn_classification/cell_0/bidirectional_rnn/fw/fw/TensorArrayUnstack/range/delta*#
_output_shapes
:˙˙˙˙˙˙˙˙˙

lrnn_classification/cell_0/bidirectional_rnn/fw/fw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3TensorArrayScatterV3?rnn_classification/cell_0/bidirectional_rnn/fw/fw/TensorArray_1Jrnn_classification/cell_0/bidirectional_rnn/fw/fw/TensorArrayUnstack/range;rnn_classification/cell_0/bidirectional_rnn/fw/fw/transposeArnn_classification/cell_0/bidirectional_rnn/fw/fw/TensorArray_1:1*
_output_shapes
: *
T0*N
_classD
B@loc:@rnn_classification/cell_0/bidirectional_rnn/fw/fw/transpose
}
;rnn_classification/cell_0/bidirectional_rnn/fw/fw/Maximum/xConst*
value	B :*
dtype0*
_output_shapes
: 
Ů
9rnn_classification/cell_0/bidirectional_rnn/fw/fw/MaximumMaximum;rnn_classification/cell_0/bidirectional_rnn/fw/fw/Maximum/x5rnn_classification/cell_0/bidirectional_rnn/fw/fw/Max*
T0*
_output_shapes
: 
á
9rnn_classification/cell_0/bidirectional_rnn/fw/fw/MinimumMinimum?rnn_classification/cell_0/bidirectional_rnn/fw/fw/strided_slice9rnn_classification/cell_0/bidirectional_rnn/fw/fw/Maximum*
_output_shapes
: *
T0

Irnn_classification/cell_0/bidirectional_rnn/fw/fw/while/iteration_counterConst*
value	B : *
dtype0*
_output_shapes
: 
¤
=rnn_classification/cell_0/bidirectional_rnn/fw/fw/while/EnterEnterIrnn_classification/cell_0/bidirectional_rnn/fw/fw/while/iteration_counter*
T0*
parallel_iterations *U

frame_nameGErnn_classification/cell_0/bidirectional_rnn/fw/fw/while/while_context*
_output_shapes
: 

?rnn_classification/cell_0/bidirectional_rnn/fw/fw/while/Enter_1Enter6rnn_classification/cell_0/bidirectional_rnn/fw/fw/time*
T0*
parallel_iterations *U

frame_nameGErnn_classification/cell_0/bidirectional_rnn/fw/fw/while/while_context*
_output_shapes
: 

?rnn_classification/cell_0/bidirectional_rnn/fw/fw/while/Enter_2Enter?rnn_classification/cell_0/bidirectional_rnn/fw/fw/TensorArray:1*
T0*
parallel_iterations *U

frame_nameGErnn_classification/cell_0/bidirectional_rnn/fw/fw/while/while_context*
_output_shapes
: 
Ě
?rnn_classification/cell_0/bidirectional_rnn/fw/fw/while/Enter_3Enterfrnn_classification/cell_0/bidirectional_rnn/fw/fw/DropoutWrapperZeroState/BasicLSTMCellZeroState/zeros*
T0*
parallel_iterations *U

frame_nameGErnn_classification/cell_0/bidirectional_rnn/fw/fw/while/while_context*
_output_shapes
:	
Î
?rnn_classification/cell_0/bidirectional_rnn/fw/fw/while/Enter_4Enterhrnn_classification/cell_0/bidirectional_rnn/fw/fw/DropoutWrapperZeroState/BasicLSTMCellZeroState/zeros_1*
T0*
parallel_iterations *U

frame_nameGErnn_classification/cell_0/bidirectional_rnn/fw/fw/while/while_context*
_output_shapes
:	
ř
=rnn_classification/cell_0/bidirectional_rnn/fw/fw/while/MergeMerge=rnn_classification/cell_0/bidirectional_rnn/fw/fw/while/EnterErnn_classification/cell_0/bidirectional_rnn/fw/fw/while/NextIteration*
T0*
N*
_output_shapes
: : 
ţ
?rnn_classification/cell_0/bidirectional_rnn/fw/fw/while/Merge_1Merge?rnn_classification/cell_0/bidirectional_rnn/fw/fw/while/Enter_1Grnn_classification/cell_0/bidirectional_rnn/fw/fw/while/NextIteration_1*
T0*
N*
_output_shapes
: : 
ţ
?rnn_classification/cell_0/bidirectional_rnn/fw/fw/while/Merge_2Merge?rnn_classification/cell_0/bidirectional_rnn/fw/fw/while/Enter_2Grnn_classification/cell_0/bidirectional_rnn/fw/fw/while/NextIteration_2*
N*
_output_shapes
: : *
T0

?rnn_classification/cell_0/bidirectional_rnn/fw/fw/while/Merge_3Merge?rnn_classification/cell_0/bidirectional_rnn/fw/fw/while/Enter_3Grnn_classification/cell_0/bidirectional_rnn/fw/fw/while/NextIteration_3*
T0*
N*!
_output_shapes
:	: 

?rnn_classification/cell_0/bidirectional_rnn/fw/fw/while/Merge_4Merge?rnn_classification/cell_0/bidirectional_rnn/fw/fw/while/Enter_4Grnn_classification/cell_0/bidirectional_rnn/fw/fw/while/NextIteration_4*
T0*
N*!
_output_shapes
:	: 
č
<rnn_classification/cell_0/bidirectional_rnn/fw/fw/while/LessLess=rnn_classification/cell_0/bidirectional_rnn/fw/fw/while/MergeBrnn_classification/cell_0/bidirectional_rnn/fw/fw/while/Less/Enter*
_output_shapes
: *
T0
˛
Brnn_classification/cell_0/bidirectional_rnn/fw/fw/while/Less/EnterEnter?rnn_classification/cell_0/bidirectional_rnn/fw/fw/strided_slice*
T0*
is_constant(*
parallel_iterations *U

frame_nameGErnn_classification/cell_0/bidirectional_rnn/fw/fw/while/while_context*
_output_shapes
: 
î
>rnn_classification/cell_0/bidirectional_rnn/fw/fw/while/Less_1Less?rnn_classification/cell_0/bidirectional_rnn/fw/fw/while/Merge_1Drnn_classification/cell_0/bidirectional_rnn/fw/fw/while/Less_1/Enter*
_output_shapes
: *
T0
Ž
Drnn_classification/cell_0/bidirectional_rnn/fw/fw/while/Less_1/EnterEnter9rnn_classification/cell_0/bidirectional_rnn/fw/fw/Minimum*
parallel_iterations *U

frame_nameGErnn_classification/cell_0/bidirectional_rnn/fw/fw/while/while_context*
_output_shapes
: *
T0*
is_constant(
ć
Brnn_classification/cell_0/bidirectional_rnn/fw/fw/while/LogicalAnd
LogicalAnd<rnn_classification/cell_0/bidirectional_rnn/fw/fw/while/Less>rnn_classification/cell_0/bidirectional_rnn/fw/fw/while/Less_1*
_output_shapes
: 
¨
@rnn_classification/cell_0/bidirectional_rnn/fw/fw/while/LoopCondLoopCondBrnn_classification/cell_0/bidirectional_rnn/fw/fw/while/LogicalAnd*
_output_shapes
: 
ž
>rnn_classification/cell_0/bidirectional_rnn/fw/fw/while/SwitchSwitch=rnn_classification/cell_0/bidirectional_rnn/fw/fw/while/Merge@rnn_classification/cell_0/bidirectional_rnn/fw/fw/while/LoopCond*
_output_shapes
: : *
T0*P
_classF
DBloc:@rnn_classification/cell_0/bidirectional_rnn/fw/fw/while/Merge
Ä
@rnn_classification/cell_0/bidirectional_rnn/fw/fw/while/Switch_1Switch?rnn_classification/cell_0/bidirectional_rnn/fw/fw/while/Merge_1@rnn_classification/cell_0/bidirectional_rnn/fw/fw/while/LoopCond*
T0*R
_classH
FDloc:@rnn_classification/cell_0/bidirectional_rnn/fw/fw/while/Merge_1*
_output_shapes
: : 
Ä
@rnn_classification/cell_0/bidirectional_rnn/fw/fw/while/Switch_2Switch?rnn_classification/cell_0/bidirectional_rnn/fw/fw/while/Merge_2@rnn_classification/cell_0/bidirectional_rnn/fw/fw/while/LoopCond*
T0*R
_classH
FDloc:@rnn_classification/cell_0/bidirectional_rnn/fw/fw/while/Merge_2*
_output_shapes
: : 
Ö
@rnn_classification/cell_0/bidirectional_rnn/fw/fw/while/Switch_3Switch?rnn_classification/cell_0/bidirectional_rnn/fw/fw/while/Merge_3@rnn_classification/cell_0/bidirectional_rnn/fw/fw/while/LoopCond*
T0*R
_classH
FDloc:@rnn_classification/cell_0/bidirectional_rnn/fw/fw/while/Merge_3**
_output_shapes
:	:	
Ö
@rnn_classification/cell_0/bidirectional_rnn/fw/fw/while/Switch_4Switch?rnn_classification/cell_0/bidirectional_rnn/fw/fw/while/Merge_4@rnn_classification/cell_0/bidirectional_rnn/fw/fw/while/LoopCond*
T0*R
_classH
FDloc:@rnn_classification/cell_0/bidirectional_rnn/fw/fw/while/Merge_4**
_output_shapes
:	:	
Ż
@rnn_classification/cell_0/bidirectional_rnn/fw/fw/while/IdentityIdentity@rnn_classification/cell_0/bidirectional_rnn/fw/fw/while/Switch:1*
_output_shapes
: *
T0
ł
Brnn_classification/cell_0/bidirectional_rnn/fw/fw/while/Identity_1IdentityBrnn_classification/cell_0/bidirectional_rnn/fw/fw/while/Switch_1:1*
_output_shapes
: *
T0
ł
Brnn_classification/cell_0/bidirectional_rnn/fw/fw/while/Identity_2IdentityBrnn_classification/cell_0/bidirectional_rnn/fw/fw/while/Switch_2:1*
T0*
_output_shapes
: 
ź
Brnn_classification/cell_0/bidirectional_rnn/fw/fw/while/Identity_3IdentityBrnn_classification/cell_0/bidirectional_rnn/fw/fw/while/Switch_3:1*
T0*
_output_shapes
:	
ź
Brnn_classification/cell_0/bidirectional_rnn/fw/fw/while/Identity_4IdentityBrnn_classification/cell_0/bidirectional_rnn/fw/fw/while/Switch_4:1*
T0*
_output_shapes
:	
Â
=rnn_classification/cell_0/bidirectional_rnn/fw/fw/while/add/yConstA^rnn_classification/cell_0/bidirectional_rnn/fw/fw/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
ä
;rnn_classification/cell_0/bidirectional_rnn/fw/fw/while/addAdd@rnn_classification/cell_0/bidirectional_rnn/fw/fw/while/Identity=rnn_classification/cell_0/bidirectional_rnn/fw/fw/while/add/y*
_output_shapes
: *
T0
ó
Irnn_classification/cell_0/bidirectional_rnn/fw/fw/while/TensorArrayReadV3TensorArrayReadV3Ornn_classification/cell_0/bidirectional_rnn/fw/fw/while/TensorArrayReadV3/EnterBrnn_classification/cell_0/bidirectional_rnn/fw/fw/while/Identity_1Qrnn_classification/cell_0/bidirectional_rnn/fw/fw/while/TensorArrayReadV3/Enter_1*
dtype0*
_output_shapes

:`
Ă
Ornn_classification/cell_0/bidirectional_rnn/fw/fw/while/TensorArrayReadV3/EnterEnter?rnn_classification/cell_0/bidirectional_rnn/fw/fw/TensorArray_1*
parallel_iterations *U

frame_nameGErnn_classification/cell_0/bidirectional_rnn/fw/fw/while/while_context*
_output_shapes
:*
T0*
is_constant(
î
Qrnn_classification/cell_0/bidirectional_rnn/fw/fw/while/TensorArrayReadV3/Enter_1Enterlrnn_classification/cell_0/bidirectional_rnn/fw/fw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3*
T0*
is_constant(*
parallel_iterations *U

frame_nameGErnn_classification/cell_0/bidirectional_rnn/fw/fw/while/while_context*
_output_shapes
: 

Drnn_classification/cell_0/bidirectional_rnn/fw/fw/while/GreaterEqualGreaterEqualBrnn_classification/cell_0/bidirectional_rnn/fw/fw/while/Identity_1Jrnn_classification/cell_0/bidirectional_rnn/fw/fw/while/GreaterEqual/Enter*
T0*
_output_shapes
:
ź
Jrnn_classification/cell_0/bidirectional_rnn/fw/fw/while/GreaterEqual/EnterEnter=rnn_classification/cell_0/bidirectional_rnn/fw/fw/CheckSeqLen*
T0*
is_constant(*
parallel_iterations *U

frame_nameGErnn_classification/cell_0/bidirectional_rnn/fw/fw/while/while_context*
_output_shapes
:

frnn_classification/cell_0/bidirectional_rnn/fw/basic_lstm_cell/kernel/Initializer/random_uniform/shapeConst*
valueB"ŕ      *X
_classN
LJloc:@rnn_classification/cell_0/bidirectional_rnn/fw/basic_lstm_cell/kernel*
dtype0*
_output_shapes
:

drnn_classification/cell_0/bidirectional_rnn/fw/basic_lstm_cell/kernel/Initializer/random_uniform/minConst*
valueB
 *Ľé¸˝*X
_classN
LJloc:@rnn_classification/cell_0/bidirectional_rnn/fw/basic_lstm_cell/kernel*
dtype0*
_output_shapes
: 

drnn_classification/cell_0/bidirectional_rnn/fw/basic_lstm_cell/kernel/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *
valueB
 *Ľé¸=*X
_classN
LJloc:@rnn_classification/cell_0/bidirectional_rnn/fw/basic_lstm_cell/kernel
ů
nrnn_classification/cell_0/bidirectional_rnn/fw/basic_lstm_cell/kernel/Initializer/random_uniform/RandomUniformRandomUniformfrnn_classification/cell_0/bidirectional_rnn/fw/basic_lstm_cell/kernel/Initializer/random_uniform/shape*
dtype0* 
_output_shapes
:
ŕ*
T0*X
_classN
LJloc:@rnn_classification/cell_0/bidirectional_rnn/fw/basic_lstm_cell/kernel
˛
drnn_classification/cell_0/bidirectional_rnn/fw/basic_lstm_cell/kernel/Initializer/random_uniform/subSubdrnn_classification/cell_0/bidirectional_rnn/fw/basic_lstm_cell/kernel/Initializer/random_uniform/maxdrnn_classification/cell_0/bidirectional_rnn/fw/basic_lstm_cell/kernel/Initializer/random_uniform/min*
T0*X
_classN
LJloc:@rnn_classification/cell_0/bidirectional_rnn/fw/basic_lstm_cell/kernel*
_output_shapes
: 
Ć
drnn_classification/cell_0/bidirectional_rnn/fw/basic_lstm_cell/kernel/Initializer/random_uniform/mulMulnrnn_classification/cell_0/bidirectional_rnn/fw/basic_lstm_cell/kernel/Initializer/random_uniform/RandomUniformdrnn_classification/cell_0/bidirectional_rnn/fw/basic_lstm_cell/kernel/Initializer/random_uniform/sub*
T0*X
_classN
LJloc:@rnn_classification/cell_0/bidirectional_rnn/fw/basic_lstm_cell/kernel* 
_output_shapes
:
ŕ
¸
`rnn_classification/cell_0/bidirectional_rnn/fw/basic_lstm_cell/kernel/Initializer/random_uniformAdddrnn_classification/cell_0/bidirectional_rnn/fw/basic_lstm_cell/kernel/Initializer/random_uniform/muldrnn_classification/cell_0/bidirectional_rnn/fw/basic_lstm_cell/kernel/Initializer/random_uniform/min*
T0*X
_classN
LJloc:@rnn_classification/cell_0/bidirectional_rnn/fw/basic_lstm_cell/kernel* 
_output_shapes
:
ŕ
ó
Ernn_classification/cell_0/bidirectional_rnn/fw/basic_lstm_cell/kernel
VariableV2*X
_classN
LJloc:@rnn_classification/cell_0/bidirectional_rnn/fw/basic_lstm_cell/kernel*
shape:
ŕ*
dtype0* 
_output_shapes
:
ŕ

Lrnn_classification/cell_0/bidirectional_rnn/fw/basic_lstm_cell/kernel/AssignAssignErnn_classification/cell_0/bidirectional_rnn/fw/basic_lstm_cell/kernel`rnn_classification/cell_0/bidirectional_rnn/fw/basic_lstm_cell/kernel/Initializer/random_uniform*
T0*X
_classN
LJloc:@rnn_classification/cell_0/bidirectional_rnn/fw/basic_lstm_cell/kernel* 
_output_shapes
:
ŕ
Č
Jrnn_classification/cell_0/bidirectional_rnn/fw/basic_lstm_cell/kernel/readIdentityErnn_classification/cell_0/bidirectional_rnn/fw/basic_lstm_cell/kernel*
T0* 
_output_shapes
:
ŕ
ü
Urnn_classification/cell_0/bidirectional_rnn/fw/basic_lstm_cell/bias/Initializer/zerosConst*
valueB*    *V
_classL
JHloc:@rnn_classification/cell_0/bidirectional_rnn/fw/basic_lstm_cell/bias*
dtype0*
_output_shapes	
:
ĺ
Crnn_classification/cell_0/bidirectional_rnn/fw/basic_lstm_cell/bias
VariableV2*
shape:*
dtype0*
_output_shapes	
:*V
_classL
JHloc:@rnn_classification/cell_0/bidirectional_rnn/fw/basic_lstm_cell/bias
î
Jrnn_classification/cell_0/bidirectional_rnn/fw/basic_lstm_cell/bias/AssignAssignCrnn_classification/cell_0/bidirectional_rnn/fw/basic_lstm_cell/biasUrnn_classification/cell_0/bidirectional_rnn/fw/basic_lstm_cell/bias/Initializer/zeros*
_output_shapes	
:*
T0*V
_classL
JHloc:@rnn_classification/cell_0/bidirectional_rnn/fw/basic_lstm_cell/bias
ż
Hrnn_classification/cell_0/bidirectional_rnn/fw/basic_lstm_cell/bias/readIdentityCrnn_classification/cell_0/bidirectional_rnn/fw/basic_lstm_cell/bias*
T0*
_output_shapes	
:
Ň
Mrnn_classification/cell_0/bidirectional_rnn/fw/fw/while/basic_lstm_cell/ConstConstA^rnn_classification/cell_0/bidirectional_rnn/fw/fw/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
Ř
Srnn_classification/cell_0/bidirectional_rnn/fw/fw/while/basic_lstm_cell/concat/axisConstA^rnn_classification/cell_0/bidirectional_rnn/fw/fw/while/Identity*
dtype0*
_output_shapes
: *
value	B :
ń
Nrnn_classification/cell_0/bidirectional_rnn/fw/fw/while/basic_lstm_cell/concatConcatV2Irnn_classification/cell_0/bidirectional_rnn/fw/fw/while/TensorArrayReadV3Brnn_classification/cell_0/bidirectional_rnn/fw/fw/while/Identity_4Srnn_classification/cell_0/bidirectional_rnn/fw/fw/while/basic_lstm_cell/concat/axis*
T0*
N*
_output_shapes
:	ŕ
¨
Nrnn_classification/cell_0/bidirectional_rnn/fw/fw/while/basic_lstm_cell/MatMulMatMulNrnn_classification/cell_0/bidirectional_rnn/fw/fw/while/basic_lstm_cell/concatTrnn_classification/cell_0/bidirectional_rnn/fw/fw/while/basic_lstm_cell/MatMul/Enter*
T0*
_output_shapes
:	
Ů
Trnn_classification/cell_0/bidirectional_rnn/fw/fw/while/basic_lstm_cell/MatMul/EnterEnterJrnn_classification/cell_0/bidirectional_rnn/fw/basic_lstm_cell/kernel/read*
parallel_iterations *U

frame_nameGErnn_classification/cell_0/bidirectional_rnn/fw/fw/while/while_context* 
_output_shapes
:
ŕ*
T0*
is_constant(
Ť
Ornn_classification/cell_0/bidirectional_rnn/fw/fw/while/basic_lstm_cell/BiasAddBiasAddNrnn_classification/cell_0/bidirectional_rnn/fw/fw/while/basic_lstm_cell/MatMulUrnn_classification/cell_0/bidirectional_rnn/fw/fw/while/basic_lstm_cell/BiasAdd/Enter*
T0*
_output_shapes
:	
Ó
Urnn_classification/cell_0/bidirectional_rnn/fw/fw/while/basic_lstm_cell/BiasAdd/EnterEnterHrnn_classification/cell_0/bidirectional_rnn/fw/basic_lstm_cell/bias/read*
T0*
is_constant(*
parallel_iterations *U

frame_nameGErnn_classification/cell_0/bidirectional_rnn/fw/fw/while/while_context*
_output_shapes	
:
Ô
Ornn_classification/cell_0/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Const_1ConstA^rnn_classification/cell_0/bidirectional_rnn/fw/fw/while/Identity*
dtype0*
_output_shapes
: *
value	B :
Ň
Mrnn_classification/cell_0/bidirectional_rnn/fw/fw/while/basic_lstm_cell/splitSplitMrnn_classification/cell_0/bidirectional_rnn/fw/fw/while/basic_lstm_cell/ConstOrnn_classification/cell_0/bidirectional_rnn/fw/fw/while/basic_lstm_cell/BiasAdd*
T0*
	num_split*@
_output_shapes.
,:	:	:	:	
×
Ornn_classification/cell_0/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Const_2ConstA^rnn_classification/cell_0/bidirectional_rnn/fw/fw/while/Identity*
valueB
 *  ?*
dtype0*
_output_shapes
: 

Krnn_classification/cell_0/bidirectional_rnn/fw/fw/while/basic_lstm_cell/AddAddOrnn_classification/cell_0/bidirectional_rnn/fw/fw/while/basic_lstm_cell/split:2Ornn_classification/cell_0/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Const_2*
T0*
_output_shapes
:	
Ń
Ornn_classification/cell_0/bidirectional_rnn/fw/fw/while/basic_lstm_cell/SigmoidSigmoidKrnn_classification/cell_0/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Add*
_output_shapes
:	*
T0

Krnn_classification/cell_0/bidirectional_rnn/fw/fw/while/basic_lstm_cell/MulMulBrnn_classification/cell_0/bidirectional_rnn/fw/fw/while/Identity_3Ornn_classification/cell_0/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Sigmoid*
T0*
_output_shapes
:	
Ő
Qrnn_classification/cell_0/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Sigmoid_1SigmoidMrnn_classification/cell_0/bidirectional_rnn/fw/fw/while/basic_lstm_cell/split*
T0*
_output_shapes
:	
Ď
Lrnn_classification/cell_0/bidirectional_rnn/fw/fw/while/basic_lstm_cell/TanhTanhOrnn_classification/cell_0/bidirectional_rnn/fw/fw/while/basic_lstm_cell/split:1*
T0*
_output_shapes
:	

Mrnn_classification/cell_0/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_1MulQrnn_classification/cell_0/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Sigmoid_1Lrnn_classification/cell_0/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Tanh*
T0*
_output_shapes
:	

Mrnn_classification/cell_0/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Add_1AddKrnn_classification/cell_0/bidirectional_rnn/fw/fw/while/basic_lstm_cell/MulMrnn_classification/cell_0/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_1*
T0*
_output_shapes
:	
Ď
Nrnn_classification/cell_0/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Tanh_1TanhMrnn_classification/cell_0/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Add_1*
T0*
_output_shapes
:	
×
Qrnn_classification/cell_0/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Sigmoid_2SigmoidOrnn_classification/cell_0/bidirectional_rnn/fw/fw/while/basic_lstm_cell/split:3*
T0*
_output_shapes
:	
Ą
Mrnn_classification/cell_0/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_2MulNrnn_classification/cell_0/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Tanh_1Qrnn_classification/cell_0/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Sigmoid_2*
T0*
_output_shapes
:	
Ż
>rnn_classification/cell_0/bidirectional_rnn/fw/fw/while/SelectSelectDrnn_classification/cell_0/bidirectional_rnn/fw/fw/while/GreaterEqualDrnn_classification/cell_0/bidirectional_rnn/fw/fw/while/Select/EnterMrnn_classification/cell_0/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_2*
T0*`
_classV
TRloc:@rnn_classification/cell_0/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_2*
_output_shapes
:	

Drnn_classification/cell_0/bidirectional_rnn/fw/fw/while/Select/EnterEnter7rnn_classification/cell_0/bidirectional_rnn/fw/fw/zeros*
is_constant(*
_output_shapes
:	*U

frame_nameGErnn_classification/cell_0/bidirectional_rnn/fw/fw/while/while_context*
T0*`
_classV
TRloc:@rnn_classification/cell_0/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_2*
parallel_iterations 
Ż
@rnn_classification/cell_0/bidirectional_rnn/fw/fw/while/Select_1SelectDrnn_classification/cell_0/bidirectional_rnn/fw/fw/while/GreaterEqualBrnn_classification/cell_0/bidirectional_rnn/fw/fw/while/Identity_3Mrnn_classification/cell_0/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Add_1*
T0*`
_classV
TRloc:@rnn_classification/cell_0/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Add_1*
_output_shapes
:	
Ż
@rnn_classification/cell_0/bidirectional_rnn/fw/fw/while/Select_2SelectDrnn_classification/cell_0/bidirectional_rnn/fw/fw/while/GreaterEqualBrnn_classification/cell_0/bidirectional_rnn/fw/fw/while/Identity_4Mrnn_classification/cell_0/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_2*
T0*`
_classV
TRloc:@rnn_classification/cell_0/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_2*
_output_shapes
:	

[rnn_classification/cell_0/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3TensorArrayWriteV3arnn_classification/cell_0/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3/EnterBrnn_classification/cell_0/bidirectional_rnn/fw/fw/while/Identity_1>rnn_classification/cell_0/bidirectional_rnn/fw/fw/while/SelectBrnn_classification/cell_0/bidirectional_rnn/fw/fw/while/Identity_2*
_output_shapes
: *
T0*`
_classV
TRloc:@rnn_classification/cell_0/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_2
ľ
arnn_classification/cell_0/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3/EnterEnter=rnn_classification/cell_0/bidirectional_rnn/fw/fw/TensorArray*U

frame_nameGErnn_classification/cell_0/bidirectional_rnn/fw/fw/while/while_context*
_output_shapes
:*
T0*`
_classV
TRloc:@rnn_classification/cell_0/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_2*
parallel_iterations *
is_constant(
Ä
?rnn_classification/cell_0/bidirectional_rnn/fw/fw/while/add_1/yConstA^rnn_classification/cell_0/bidirectional_rnn/fw/fw/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
ę
=rnn_classification/cell_0/bidirectional_rnn/fw/fw/while/add_1AddBrnn_classification/cell_0/bidirectional_rnn/fw/fw/while/Identity_1?rnn_classification/cell_0/bidirectional_rnn/fw/fw/while/add_1/y*
_output_shapes
: *
T0
´
Ernn_classification/cell_0/bidirectional_rnn/fw/fw/while/NextIterationNextIteration;rnn_classification/cell_0/bidirectional_rnn/fw/fw/while/add*
T0*
_output_shapes
: 
¸
Grnn_classification/cell_0/bidirectional_rnn/fw/fw/while/NextIteration_1NextIteration=rnn_classification/cell_0/bidirectional_rnn/fw/fw/while/add_1*
T0*
_output_shapes
: 
Ö
Grnn_classification/cell_0/bidirectional_rnn/fw/fw/while/NextIteration_2NextIteration[rnn_classification/cell_0/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3*
T0*
_output_shapes
: 
Ä
Grnn_classification/cell_0/bidirectional_rnn/fw/fw/while/NextIteration_3NextIteration@rnn_classification/cell_0/bidirectional_rnn/fw/fw/while/Select_1*
T0*
_output_shapes
:	
Ä
Grnn_classification/cell_0/bidirectional_rnn/fw/fw/while/NextIteration_4NextIteration@rnn_classification/cell_0/bidirectional_rnn/fw/fw/while/Select_2*
T0*
_output_shapes
:	
Ľ
<rnn_classification/cell_0/bidirectional_rnn/fw/fw/while/ExitExit>rnn_classification/cell_0/bidirectional_rnn/fw/fw/while/Switch*
_output_shapes
: *
T0
Š
>rnn_classification/cell_0/bidirectional_rnn/fw/fw/while/Exit_1Exit@rnn_classification/cell_0/bidirectional_rnn/fw/fw/while/Switch_1*
T0*
_output_shapes
: 
Š
>rnn_classification/cell_0/bidirectional_rnn/fw/fw/while/Exit_2Exit@rnn_classification/cell_0/bidirectional_rnn/fw/fw/while/Switch_2*
_output_shapes
: *
T0
˛
>rnn_classification/cell_0/bidirectional_rnn/fw/fw/while/Exit_3Exit@rnn_classification/cell_0/bidirectional_rnn/fw/fw/while/Switch_3*
T0*
_output_shapes
:	
˛
>rnn_classification/cell_0/bidirectional_rnn/fw/fw/while/Exit_4Exit@rnn_classification/cell_0/bidirectional_rnn/fw/fw/while/Switch_4*
T0*
_output_shapes
:	
Ň
Trnn_classification/cell_0/bidirectional_rnn/fw/fw/TensorArrayStack/TensorArraySizeV3TensorArraySizeV3=rnn_classification/cell_0/bidirectional_rnn/fw/fw/TensorArray>rnn_classification/cell_0/bidirectional_rnn/fw/fw/while/Exit_2*P
_classF
DBloc:@rnn_classification/cell_0/bidirectional_rnn/fw/fw/TensorArray*
_output_shapes
: 
â
Nrnn_classification/cell_0/bidirectional_rnn/fw/fw/TensorArrayStack/range/startConst*
dtype0*
_output_shapes
: *
value	B : *P
_classF
DBloc:@rnn_classification/cell_0/bidirectional_rnn/fw/fw/TensorArray
â
Nrnn_classification/cell_0/bidirectional_rnn/fw/fw/TensorArrayStack/range/deltaConst*
value	B :*P
_classF
DBloc:@rnn_classification/cell_0/bidirectional_rnn/fw/fw/TensorArray*
dtype0*
_output_shapes
: 
ž
Hrnn_classification/cell_0/bidirectional_rnn/fw/fw/TensorArrayStack/rangeRangeNrnn_classification/cell_0/bidirectional_rnn/fw/fw/TensorArrayStack/range/startTrnn_classification/cell_0/bidirectional_rnn/fw/fw/TensorArrayStack/TensorArraySizeV3Nrnn_classification/cell_0/bidirectional_rnn/fw/fw/TensorArrayStack/range/delta*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*P
_classF
DBloc:@rnn_classification/cell_0/bidirectional_rnn/fw/fw/TensorArray
á
Vrnn_classification/cell_0/bidirectional_rnn/fw/fw/TensorArrayStack/TensorArrayGatherV3TensorArrayGatherV3=rnn_classification/cell_0/bidirectional_rnn/fw/fw/TensorArrayHrnn_classification/cell_0/bidirectional_rnn/fw/fw/TensorArrayStack/range>rnn_classification/cell_0/bidirectional_rnn/fw/fw/while/Exit_2*P
_classF
DBloc:@rnn_classification/cell_0/bidirectional_rnn/fw/fw/TensorArray*
dtype0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙*
element_shape:	

9rnn_classification/cell_0/bidirectional_rnn/fw/fw/Const_5Const*
dtype0*
_output_shapes
:*
valueB:
z
8rnn_classification/cell_0/bidirectional_rnn/fw/fw/Rank_1Const*
value	B :*
dtype0*
_output_shapes
: 

?rnn_classification/cell_0/bidirectional_rnn/fw/fw/range_1/startConst*
value	B :*
dtype0*
_output_shapes
: 

?rnn_classification/cell_0/bidirectional_rnn/fw/fw/range_1/deltaConst*
value	B :*
dtype0*
_output_shapes
: 

9rnn_classification/cell_0/bidirectional_rnn/fw/fw/range_1Range?rnn_classification/cell_0/bidirectional_rnn/fw/fw/range_1/start8rnn_classification/cell_0/bidirectional_rnn/fw/fw/Rank_1?rnn_classification/cell_0/bidirectional_rnn/fw/fw/range_1/delta*
_output_shapes
:

Crnn_classification/cell_0/bidirectional_rnn/fw/fw/concat_2/values_0Const*
valueB"       *
dtype0*
_output_shapes
:

?rnn_classification/cell_0/bidirectional_rnn/fw/fw/concat_2/axisConst*
dtype0*
_output_shapes
: *
value	B : 
ľ
:rnn_classification/cell_0/bidirectional_rnn/fw/fw/concat_2ConcatV2Crnn_classification/cell_0/bidirectional_rnn/fw/fw/concat_2/values_09rnn_classification/cell_0/bidirectional_rnn/fw/fw/range_1?rnn_classification/cell_0/bidirectional_rnn/fw/fw/concat_2/axis*
T0*
N*
_output_shapes
:

=rnn_classification/cell_0/bidirectional_rnn/fw/fw/transpose_1	TransposeVrnn_classification/cell_0/bidirectional_rnn/fw/fw/TensorArrayStack/TensorArrayGatherV3:rnn_classification/cell_0/bidirectional_rnn/fw/fw/concat_2*,
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
ą
>rnn_classification/cell_0/bidirectional_rnn/bw/ReverseSequenceReverseSequenceconv1d_2/BiasAddSqueeze*
T0*
seq_dim*+
_output_shapes
:˙˙˙˙˙˙˙˙˙`
x
6rnn_classification/cell_0/bidirectional_rnn/bw/bw/RankConst*
value	B :*
dtype0*
_output_shapes
: 

=rnn_classification/cell_0/bidirectional_rnn/bw/bw/range/startConst*
dtype0*
_output_shapes
: *
value	B :

=rnn_classification/cell_0/bidirectional_rnn/bw/bw/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 

7rnn_classification/cell_0/bidirectional_rnn/bw/bw/rangeRange=rnn_classification/cell_0/bidirectional_rnn/bw/bw/range/start6rnn_classification/cell_0/bidirectional_rnn/bw/bw/Rank=rnn_classification/cell_0/bidirectional_rnn/bw/bw/range/delta*
_output_shapes
:

Arnn_classification/cell_0/bidirectional_rnn/bw/bw/concat/values_0Const*
dtype0*
_output_shapes
:*
valueB"       

=rnn_classification/cell_0/bidirectional_rnn/bw/bw/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
­
8rnn_classification/cell_0/bidirectional_rnn/bw/bw/concatConcatV2Arnn_classification/cell_0/bidirectional_rnn/bw/bw/concat/values_07rnn_classification/cell_0/bidirectional_rnn/bw/bw/range=rnn_classification/cell_0/bidirectional_rnn/bw/bw/concat/axis*
T0*
N*
_output_shapes
:
ř
;rnn_classification/cell_0/bidirectional_rnn/bw/bw/transpose	Transpose>rnn_classification/cell_0/bidirectional_rnn/bw/ReverseSequence8rnn_classification/cell_0/bidirectional_rnn/bw/bw/concat*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙`
~
9rnn_classification/cell_0/bidirectional_rnn/bw/bw/ToInt32CastSqueeze*

SrcT0	*

DstT0*
_output_shapes
:
­
Arnn_classification/cell_0/bidirectional_rnn/bw/bw/sequence_lengthIdentity9rnn_classification/cell_0/bidirectional_rnn/bw/bw/ToInt32*
T0*
_output_shapes
:
°
frnn_classification/cell_0/bidirectional_rnn/bw/bw/DropoutWrapperZeroState/BasicLSTMCellZeroState/ConstConst*
valueB:*
dtype0*
_output_shapes
:
ł
hrnn_classification/cell_0/bidirectional_rnn/bw/bw/DropoutWrapperZeroState/BasicLSTMCellZeroState/Const_1Const*
valueB:*
dtype0*
_output_shapes
:
Ž
lrnn_classification/cell_0/bidirectional_rnn/bw/bw/DropoutWrapperZeroState/BasicLSTMCellZeroState/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
á
grnn_classification/cell_0/bidirectional_rnn/bw/bw/DropoutWrapperZeroState/BasicLSTMCellZeroState/concatConcatV2frnn_classification/cell_0/bidirectional_rnn/bw/bw/DropoutWrapperZeroState/BasicLSTMCellZeroState/Consthrnn_classification/cell_0/bidirectional_rnn/bw/bw/DropoutWrapperZeroState/BasicLSTMCellZeroState/Const_1lrnn_classification/cell_0/bidirectional_rnn/bw/bw/DropoutWrapperZeroState/BasicLSTMCellZeroState/concat/axis*
T0*
N*
_output_shapes
:
ą
lrnn_classification/cell_0/bidirectional_rnn/bw/bw/DropoutWrapperZeroState/BasicLSTMCellZeroState/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
ď
frnn_classification/cell_0/bidirectional_rnn/bw/bw/DropoutWrapperZeroState/BasicLSTMCellZeroState/zerosFillgrnn_classification/cell_0/bidirectional_rnn/bw/bw/DropoutWrapperZeroState/BasicLSTMCellZeroState/concatlrnn_classification/cell_0/bidirectional_rnn/bw/bw/DropoutWrapperZeroState/BasicLSTMCellZeroState/zeros/Const*
T0*
_output_shapes
:	
˛
hrnn_classification/cell_0/bidirectional_rnn/bw/bw/DropoutWrapperZeroState/BasicLSTMCellZeroState/Const_2Const*
valueB:*
dtype0*
_output_shapes
:
ł
hrnn_classification/cell_0/bidirectional_rnn/bw/bw/DropoutWrapperZeroState/BasicLSTMCellZeroState/Const_3Const*
dtype0*
_output_shapes
:*
valueB:
˛
hrnn_classification/cell_0/bidirectional_rnn/bw/bw/DropoutWrapperZeroState/BasicLSTMCellZeroState/Const_4Const*
dtype0*
_output_shapes
:*
valueB:
ł
hrnn_classification/cell_0/bidirectional_rnn/bw/bw/DropoutWrapperZeroState/BasicLSTMCellZeroState/Const_5Const*
valueB:*
dtype0*
_output_shapes
:
°
nrnn_classification/cell_0/bidirectional_rnn/bw/bw/DropoutWrapperZeroState/BasicLSTMCellZeroState/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
ç
irnn_classification/cell_0/bidirectional_rnn/bw/bw/DropoutWrapperZeroState/BasicLSTMCellZeroState/concat_1ConcatV2hrnn_classification/cell_0/bidirectional_rnn/bw/bw/DropoutWrapperZeroState/BasicLSTMCellZeroState/Const_4hrnn_classification/cell_0/bidirectional_rnn/bw/bw/DropoutWrapperZeroState/BasicLSTMCellZeroState/Const_5nrnn_classification/cell_0/bidirectional_rnn/bw/bw/DropoutWrapperZeroState/BasicLSTMCellZeroState/concat_1/axis*
T0*
N*
_output_shapes
:
ł
nrnn_classification/cell_0/bidirectional_rnn/bw/bw/DropoutWrapperZeroState/BasicLSTMCellZeroState/zeros_1/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
ő
hrnn_classification/cell_0/bidirectional_rnn/bw/bw/DropoutWrapperZeroState/BasicLSTMCellZeroState/zeros_1Fillirnn_classification/cell_0/bidirectional_rnn/bw/bw/DropoutWrapperZeroState/BasicLSTMCellZeroState/concat_1nrnn_classification/cell_0/bidirectional_rnn/bw/bw/DropoutWrapperZeroState/BasicLSTMCellZeroState/zeros_1/Const*
T0*
_output_shapes
:	
˛
hrnn_classification/cell_0/bidirectional_rnn/bw/bw/DropoutWrapperZeroState/BasicLSTMCellZeroState/Const_6Const*
dtype0*
_output_shapes
:*
valueB:
ł
hrnn_classification/cell_0/bidirectional_rnn/bw/bw/DropoutWrapperZeroState/BasicLSTMCellZeroState/Const_7Const*
dtype0*
_output_shapes
:*
valueB:

7rnn_classification/cell_0/bidirectional_rnn/bw/bw/ShapeConst*
valueB:*
dtype0*
_output_shapes
:

7rnn_classification/cell_0/bidirectional_rnn/bw/bw/stackConst*
valueB:*
dtype0*
_output_shapes
:
×
7rnn_classification/cell_0/bidirectional_rnn/bw/bw/EqualEqual7rnn_classification/cell_0/bidirectional_rnn/bw/bw/Shape7rnn_classification/cell_0/bidirectional_rnn/bw/bw/stack*
T0*
_output_shapes
:

7rnn_classification/cell_0/bidirectional_rnn/bw/bw/ConstConst*
dtype0*
_output_shapes
:*
valueB: 
Ć
5rnn_classification/cell_0/bidirectional_rnn/bw/bw/AllAll7rnn_classification/cell_0/bidirectional_rnn/bw/bw/Equal7rnn_classification/cell_0/bidirectional_rnn/bw/bw/Const*
_output_shapes
: 
ŕ
>rnn_classification/cell_0/bidirectional_rnn/bw/bw/Assert/ConstConst*r
valueiBg BaExpected shape for Tensor rnn_classification/cell_0/bidirectional_rnn/bw/bw/sequence_length:0 is *
dtype0*
_output_shapes
: 

@rnn_classification/cell_0/bidirectional_rnn/bw/bw/Assert/Const_1Const*!
valueB B but saw shape: *
dtype0*
_output_shapes
: 
č
Frnn_classification/cell_0/bidirectional_rnn/bw/bw/Assert/Assert/data_0Const*r
valueiBg BaExpected shape for Tensor rnn_classification/cell_0/bidirectional_rnn/bw/bw/sequence_length:0 is *
dtype0*
_output_shapes
: 

Frnn_classification/cell_0/bidirectional_rnn/bw/bw/Assert/Assert/data_2Const*
dtype0*
_output_shapes
: *!
valueB B but saw shape: 

?rnn_classification/cell_0/bidirectional_rnn/bw/bw/Assert/AssertAssert5rnn_classification/cell_0/bidirectional_rnn/bw/bw/AllFrnn_classification/cell_0/bidirectional_rnn/bw/bw/Assert/Assert/data_07rnn_classification/cell_0/bidirectional_rnn/bw/bw/stackFrnn_classification/cell_0/bidirectional_rnn/bw/bw/Assert/Assert/data_27rnn_classification/cell_0/bidirectional_rnn/bw/bw/Shape*
T
2
ó
=rnn_classification/cell_0/bidirectional_rnn/bw/bw/CheckSeqLenIdentityArnn_classification/cell_0/bidirectional_rnn/bw/bw/sequence_length@^rnn_classification/cell_0/bidirectional_rnn/bw/bw/Assert/Assert*
T0*
_output_shapes
:
¤
9rnn_classification/cell_0/bidirectional_rnn/bw/bw/Shape_1Shape;rnn_classification/cell_0/bidirectional_rnn/bw/bw/transpose*
_output_shapes
:*
T0

Ernn_classification/cell_0/bidirectional_rnn/bw/bw/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:

Grnn_classification/cell_0/bidirectional_rnn/bw/bw/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:

Grnn_classification/cell_0/bidirectional_rnn/bw/bw/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
Š
?rnn_classification/cell_0/bidirectional_rnn/bw/bw/strided_sliceStridedSlice9rnn_classification/cell_0/bidirectional_rnn/bw/bw/Shape_1Ernn_classification/cell_0/bidirectional_rnn/bw/bw/strided_slice/stackGrnn_classification/cell_0/bidirectional_rnn/bw/bw/strided_slice/stack_1Grnn_classification/cell_0/bidirectional_rnn/bw/bw/strided_slice/stack_2*
shrink_axis_mask*
_output_shapes
: *
Index0*
T0

9rnn_classification/cell_0/bidirectional_rnn/bw/bw/Const_1Const*
valueB:*
dtype0*
_output_shapes
:

9rnn_classification/cell_0/bidirectional_rnn/bw/bw/Const_2Const*
valueB:*
dtype0*
_output_shapes
:

?rnn_classification/cell_0/bidirectional_rnn/bw/bw/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
Ť
:rnn_classification/cell_0/bidirectional_rnn/bw/bw/concat_1ConcatV29rnn_classification/cell_0/bidirectional_rnn/bw/bw/Const_19rnn_classification/cell_0/bidirectional_rnn/bw/bw/Const_2?rnn_classification/cell_0/bidirectional_rnn/bw/bw/concat_1/axis*
T0*
N*
_output_shapes
:

=rnn_classification/cell_0/bidirectional_rnn/bw/bw/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
ä
7rnn_classification/cell_0/bidirectional_rnn/bw/bw/zerosFill:rnn_classification/cell_0/bidirectional_rnn/bw/bw/concat_1=rnn_classification/cell_0/bidirectional_rnn/bw/bw/zeros/Const*
T0*
_output_shapes
:	

9rnn_classification/cell_0/bidirectional_rnn/bw/bw/Const_3Const*
valueB: *
dtype0*
_output_shapes
:
×
5rnn_classification/cell_0/bidirectional_rnn/bw/bw/MinMin=rnn_classification/cell_0/bidirectional_rnn/bw/bw/CheckSeqLen9rnn_classification/cell_0/bidirectional_rnn/bw/bw/Const_3*
T0*
_output_shapes
: 

9rnn_classification/cell_0/bidirectional_rnn/bw/bw/Const_4Const*
valueB: *
dtype0*
_output_shapes
:
×
5rnn_classification/cell_0/bidirectional_rnn/bw/bw/MaxMax=rnn_classification/cell_0/bidirectional_rnn/bw/bw/CheckSeqLen9rnn_classification/cell_0/bidirectional_rnn/bw/bw/Const_4*
T0*
_output_shapes
: 
x
6rnn_classification/cell_0/bidirectional_rnn/bw/bw/timeConst*
value	B : *
dtype0*
_output_shapes
: 
×
=rnn_classification/cell_0/bidirectional_rnn/bw/bw/TensorArrayTensorArrayV3?rnn_classification/cell_0/bidirectional_rnn/bw/bw/strided_slice*]
tensor_array_nameHFrnn_classification/cell_0/bidirectional_rnn/bw/bw/dynamic_rnn/output_0*
dtype0*
_output_shapes

:: *
element_shape:	*
identical_element_shapes(
×
?rnn_classification/cell_0/bidirectional_rnn/bw/bw/TensorArray_1TensorArrayV3?rnn_classification/cell_0/bidirectional_rnn/bw/bw/strided_slice*\
tensor_array_nameGErnn_classification/cell_0/bidirectional_rnn/bw/bw/dynamic_rnn/input_0*
dtype0*
_output_shapes

:: *
element_shape
:`*
identical_element_shapes(
ľ
Jrnn_classification/cell_0/bidirectional_rnn/bw/bw/TensorArrayUnstack/ShapeShape;rnn_classification/cell_0/bidirectional_rnn/bw/bw/transpose*
T0*
_output_shapes
:
˘
Xrnn_classification/cell_0/bidirectional_rnn/bw/bw/TensorArrayUnstack/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
¤
Zrnn_classification/cell_0/bidirectional_rnn/bw/bw/TensorArrayUnstack/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
¤
Zrnn_classification/cell_0/bidirectional_rnn/bw/bw/TensorArrayUnstack/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:

Rrnn_classification/cell_0/bidirectional_rnn/bw/bw/TensorArrayUnstack/strided_sliceStridedSliceJrnn_classification/cell_0/bidirectional_rnn/bw/bw/TensorArrayUnstack/ShapeXrnn_classification/cell_0/bidirectional_rnn/bw/bw/TensorArrayUnstack/strided_slice/stackZrnn_classification/cell_0/bidirectional_rnn/bw/bw/TensorArrayUnstack/strided_slice/stack_1Zrnn_classification/cell_0/bidirectional_rnn/bw/bw/TensorArrayUnstack/strided_slice/stack_2*
Index0*
T0*
shrink_axis_mask*
_output_shapes
: 

Prnn_classification/cell_0/bidirectional_rnn/bw/bw/TensorArrayUnstack/range/startConst*
value	B : *
dtype0*
_output_shapes
: 

Prnn_classification/cell_0/bidirectional_rnn/bw/bw/TensorArrayUnstack/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
đ
Jrnn_classification/cell_0/bidirectional_rnn/bw/bw/TensorArrayUnstack/rangeRangePrnn_classification/cell_0/bidirectional_rnn/bw/bw/TensorArrayUnstack/range/startRrnn_classification/cell_0/bidirectional_rnn/bw/bw/TensorArrayUnstack/strided_slicePrnn_classification/cell_0/bidirectional_rnn/bw/bw/TensorArrayUnstack/range/delta*#
_output_shapes
:˙˙˙˙˙˙˙˙˙

lrnn_classification/cell_0/bidirectional_rnn/bw/bw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3TensorArrayScatterV3?rnn_classification/cell_0/bidirectional_rnn/bw/bw/TensorArray_1Jrnn_classification/cell_0/bidirectional_rnn/bw/bw/TensorArrayUnstack/range;rnn_classification/cell_0/bidirectional_rnn/bw/bw/transposeArnn_classification/cell_0/bidirectional_rnn/bw/bw/TensorArray_1:1*
T0*N
_classD
B@loc:@rnn_classification/cell_0/bidirectional_rnn/bw/bw/transpose*
_output_shapes
: 
}
;rnn_classification/cell_0/bidirectional_rnn/bw/bw/Maximum/xConst*
value	B :*
dtype0*
_output_shapes
: 
Ů
9rnn_classification/cell_0/bidirectional_rnn/bw/bw/MaximumMaximum;rnn_classification/cell_0/bidirectional_rnn/bw/bw/Maximum/x5rnn_classification/cell_0/bidirectional_rnn/bw/bw/Max*
T0*
_output_shapes
: 
á
9rnn_classification/cell_0/bidirectional_rnn/bw/bw/MinimumMinimum?rnn_classification/cell_0/bidirectional_rnn/bw/bw/strided_slice9rnn_classification/cell_0/bidirectional_rnn/bw/bw/Maximum*
T0*
_output_shapes
: 

Irnn_classification/cell_0/bidirectional_rnn/bw/bw/while/iteration_counterConst*
value	B : *
dtype0*
_output_shapes
: 
¤
=rnn_classification/cell_0/bidirectional_rnn/bw/bw/while/EnterEnterIrnn_classification/cell_0/bidirectional_rnn/bw/bw/while/iteration_counter*
T0*
parallel_iterations *U

frame_nameGErnn_classification/cell_0/bidirectional_rnn/bw/bw/while/while_context*
_output_shapes
: 

?rnn_classification/cell_0/bidirectional_rnn/bw/bw/while/Enter_1Enter6rnn_classification/cell_0/bidirectional_rnn/bw/bw/time*
T0*
parallel_iterations *U

frame_nameGErnn_classification/cell_0/bidirectional_rnn/bw/bw/while/while_context*
_output_shapes
: 

?rnn_classification/cell_0/bidirectional_rnn/bw/bw/while/Enter_2Enter?rnn_classification/cell_0/bidirectional_rnn/bw/bw/TensorArray:1*
T0*
parallel_iterations *U

frame_nameGErnn_classification/cell_0/bidirectional_rnn/bw/bw/while/while_context*
_output_shapes
: 
Ě
?rnn_classification/cell_0/bidirectional_rnn/bw/bw/while/Enter_3Enterfrnn_classification/cell_0/bidirectional_rnn/bw/bw/DropoutWrapperZeroState/BasicLSTMCellZeroState/zeros*
T0*
parallel_iterations *U

frame_nameGErnn_classification/cell_0/bidirectional_rnn/bw/bw/while/while_context*
_output_shapes
:	
Î
?rnn_classification/cell_0/bidirectional_rnn/bw/bw/while/Enter_4Enterhrnn_classification/cell_0/bidirectional_rnn/bw/bw/DropoutWrapperZeroState/BasicLSTMCellZeroState/zeros_1*
T0*
parallel_iterations *U

frame_nameGErnn_classification/cell_0/bidirectional_rnn/bw/bw/while/while_context*
_output_shapes
:	
ř
=rnn_classification/cell_0/bidirectional_rnn/bw/bw/while/MergeMerge=rnn_classification/cell_0/bidirectional_rnn/bw/bw/while/EnterErnn_classification/cell_0/bidirectional_rnn/bw/bw/while/NextIteration*
N*
_output_shapes
: : *
T0
ţ
?rnn_classification/cell_0/bidirectional_rnn/bw/bw/while/Merge_1Merge?rnn_classification/cell_0/bidirectional_rnn/bw/bw/while/Enter_1Grnn_classification/cell_0/bidirectional_rnn/bw/bw/while/NextIteration_1*
T0*
N*
_output_shapes
: : 
ţ
?rnn_classification/cell_0/bidirectional_rnn/bw/bw/while/Merge_2Merge?rnn_classification/cell_0/bidirectional_rnn/bw/bw/while/Enter_2Grnn_classification/cell_0/bidirectional_rnn/bw/bw/while/NextIteration_2*
T0*
N*
_output_shapes
: : 

?rnn_classification/cell_0/bidirectional_rnn/bw/bw/while/Merge_3Merge?rnn_classification/cell_0/bidirectional_rnn/bw/bw/while/Enter_3Grnn_classification/cell_0/bidirectional_rnn/bw/bw/while/NextIteration_3*
N*!
_output_shapes
:	: *
T0

?rnn_classification/cell_0/bidirectional_rnn/bw/bw/while/Merge_4Merge?rnn_classification/cell_0/bidirectional_rnn/bw/bw/while/Enter_4Grnn_classification/cell_0/bidirectional_rnn/bw/bw/while/NextIteration_4*
T0*
N*!
_output_shapes
:	: 
č
<rnn_classification/cell_0/bidirectional_rnn/bw/bw/while/LessLess=rnn_classification/cell_0/bidirectional_rnn/bw/bw/while/MergeBrnn_classification/cell_0/bidirectional_rnn/bw/bw/while/Less/Enter*
T0*
_output_shapes
: 
˛
Brnn_classification/cell_0/bidirectional_rnn/bw/bw/while/Less/EnterEnter?rnn_classification/cell_0/bidirectional_rnn/bw/bw/strided_slice*
T0*
is_constant(*
parallel_iterations *U

frame_nameGErnn_classification/cell_0/bidirectional_rnn/bw/bw/while/while_context*
_output_shapes
: 
î
>rnn_classification/cell_0/bidirectional_rnn/bw/bw/while/Less_1Less?rnn_classification/cell_0/bidirectional_rnn/bw/bw/while/Merge_1Drnn_classification/cell_0/bidirectional_rnn/bw/bw/while/Less_1/Enter*
_output_shapes
: *
T0
Ž
Drnn_classification/cell_0/bidirectional_rnn/bw/bw/while/Less_1/EnterEnter9rnn_classification/cell_0/bidirectional_rnn/bw/bw/Minimum*
T0*
is_constant(*
parallel_iterations *U

frame_nameGErnn_classification/cell_0/bidirectional_rnn/bw/bw/while/while_context*
_output_shapes
: 
ć
Brnn_classification/cell_0/bidirectional_rnn/bw/bw/while/LogicalAnd
LogicalAnd<rnn_classification/cell_0/bidirectional_rnn/bw/bw/while/Less>rnn_classification/cell_0/bidirectional_rnn/bw/bw/while/Less_1*
_output_shapes
: 
¨
@rnn_classification/cell_0/bidirectional_rnn/bw/bw/while/LoopCondLoopCondBrnn_classification/cell_0/bidirectional_rnn/bw/bw/while/LogicalAnd*
_output_shapes
: 
ž
>rnn_classification/cell_0/bidirectional_rnn/bw/bw/while/SwitchSwitch=rnn_classification/cell_0/bidirectional_rnn/bw/bw/while/Merge@rnn_classification/cell_0/bidirectional_rnn/bw/bw/while/LoopCond*
T0*P
_classF
DBloc:@rnn_classification/cell_0/bidirectional_rnn/bw/bw/while/Merge*
_output_shapes
: : 
Ä
@rnn_classification/cell_0/bidirectional_rnn/bw/bw/while/Switch_1Switch?rnn_classification/cell_0/bidirectional_rnn/bw/bw/while/Merge_1@rnn_classification/cell_0/bidirectional_rnn/bw/bw/while/LoopCond*
_output_shapes
: : *
T0*R
_classH
FDloc:@rnn_classification/cell_0/bidirectional_rnn/bw/bw/while/Merge_1
Ä
@rnn_classification/cell_0/bidirectional_rnn/bw/bw/while/Switch_2Switch?rnn_classification/cell_0/bidirectional_rnn/bw/bw/while/Merge_2@rnn_classification/cell_0/bidirectional_rnn/bw/bw/while/LoopCond*
T0*R
_classH
FDloc:@rnn_classification/cell_0/bidirectional_rnn/bw/bw/while/Merge_2*
_output_shapes
: : 
Ö
@rnn_classification/cell_0/bidirectional_rnn/bw/bw/while/Switch_3Switch?rnn_classification/cell_0/bidirectional_rnn/bw/bw/while/Merge_3@rnn_classification/cell_0/bidirectional_rnn/bw/bw/while/LoopCond*
T0*R
_classH
FDloc:@rnn_classification/cell_0/bidirectional_rnn/bw/bw/while/Merge_3**
_output_shapes
:	:	
Ö
@rnn_classification/cell_0/bidirectional_rnn/bw/bw/while/Switch_4Switch?rnn_classification/cell_0/bidirectional_rnn/bw/bw/while/Merge_4@rnn_classification/cell_0/bidirectional_rnn/bw/bw/while/LoopCond*
T0*R
_classH
FDloc:@rnn_classification/cell_0/bidirectional_rnn/bw/bw/while/Merge_4**
_output_shapes
:	:	
Ż
@rnn_classification/cell_0/bidirectional_rnn/bw/bw/while/IdentityIdentity@rnn_classification/cell_0/bidirectional_rnn/bw/bw/while/Switch:1*
T0*
_output_shapes
: 
ł
Brnn_classification/cell_0/bidirectional_rnn/bw/bw/while/Identity_1IdentityBrnn_classification/cell_0/bidirectional_rnn/bw/bw/while/Switch_1:1*
T0*
_output_shapes
: 
ł
Brnn_classification/cell_0/bidirectional_rnn/bw/bw/while/Identity_2IdentityBrnn_classification/cell_0/bidirectional_rnn/bw/bw/while/Switch_2:1*
T0*
_output_shapes
: 
ź
Brnn_classification/cell_0/bidirectional_rnn/bw/bw/while/Identity_3IdentityBrnn_classification/cell_0/bidirectional_rnn/bw/bw/while/Switch_3:1*
_output_shapes
:	*
T0
ź
Brnn_classification/cell_0/bidirectional_rnn/bw/bw/while/Identity_4IdentityBrnn_classification/cell_0/bidirectional_rnn/bw/bw/while/Switch_4:1*
T0*
_output_shapes
:	
Â
=rnn_classification/cell_0/bidirectional_rnn/bw/bw/while/add/yConstA^rnn_classification/cell_0/bidirectional_rnn/bw/bw/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
ä
;rnn_classification/cell_0/bidirectional_rnn/bw/bw/while/addAdd@rnn_classification/cell_0/bidirectional_rnn/bw/bw/while/Identity=rnn_classification/cell_0/bidirectional_rnn/bw/bw/while/add/y*
T0*
_output_shapes
: 
ó
Irnn_classification/cell_0/bidirectional_rnn/bw/bw/while/TensorArrayReadV3TensorArrayReadV3Ornn_classification/cell_0/bidirectional_rnn/bw/bw/while/TensorArrayReadV3/EnterBrnn_classification/cell_0/bidirectional_rnn/bw/bw/while/Identity_1Qrnn_classification/cell_0/bidirectional_rnn/bw/bw/while/TensorArrayReadV3/Enter_1*
dtype0*
_output_shapes

:`
Ă
Ornn_classification/cell_0/bidirectional_rnn/bw/bw/while/TensorArrayReadV3/EnterEnter?rnn_classification/cell_0/bidirectional_rnn/bw/bw/TensorArray_1*
T0*
is_constant(*
parallel_iterations *U

frame_nameGErnn_classification/cell_0/bidirectional_rnn/bw/bw/while/while_context*
_output_shapes
:
î
Qrnn_classification/cell_0/bidirectional_rnn/bw/bw/while/TensorArrayReadV3/Enter_1Enterlrnn_classification/cell_0/bidirectional_rnn/bw/bw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3*
parallel_iterations *U

frame_nameGErnn_classification/cell_0/bidirectional_rnn/bw/bw/while/while_context*
_output_shapes
: *
T0*
is_constant(

Drnn_classification/cell_0/bidirectional_rnn/bw/bw/while/GreaterEqualGreaterEqualBrnn_classification/cell_0/bidirectional_rnn/bw/bw/while/Identity_1Jrnn_classification/cell_0/bidirectional_rnn/bw/bw/while/GreaterEqual/Enter*
T0*
_output_shapes
:
ź
Jrnn_classification/cell_0/bidirectional_rnn/bw/bw/while/GreaterEqual/EnterEnter=rnn_classification/cell_0/bidirectional_rnn/bw/bw/CheckSeqLen*
T0*
is_constant(*
parallel_iterations *U

frame_nameGErnn_classification/cell_0/bidirectional_rnn/bw/bw/while/while_context*
_output_shapes
:

frnn_classification/cell_0/bidirectional_rnn/bw/basic_lstm_cell/kernel/Initializer/random_uniform/shapeConst*
valueB"ŕ      *X
_classN
LJloc:@rnn_classification/cell_0/bidirectional_rnn/bw/basic_lstm_cell/kernel*
dtype0*
_output_shapes
:

drnn_classification/cell_0/bidirectional_rnn/bw/basic_lstm_cell/kernel/Initializer/random_uniform/minConst*
valueB
 *Ľé¸˝*X
_classN
LJloc:@rnn_classification/cell_0/bidirectional_rnn/bw/basic_lstm_cell/kernel*
dtype0*
_output_shapes
: 

drnn_classification/cell_0/bidirectional_rnn/bw/basic_lstm_cell/kernel/Initializer/random_uniform/maxConst*
valueB
 *Ľé¸=*X
_classN
LJloc:@rnn_classification/cell_0/bidirectional_rnn/bw/basic_lstm_cell/kernel*
dtype0*
_output_shapes
: 
ů
nrnn_classification/cell_0/bidirectional_rnn/bw/basic_lstm_cell/kernel/Initializer/random_uniform/RandomUniformRandomUniformfrnn_classification/cell_0/bidirectional_rnn/bw/basic_lstm_cell/kernel/Initializer/random_uniform/shape*
T0*X
_classN
LJloc:@rnn_classification/cell_0/bidirectional_rnn/bw/basic_lstm_cell/kernel*
dtype0* 
_output_shapes
:
ŕ
˛
drnn_classification/cell_0/bidirectional_rnn/bw/basic_lstm_cell/kernel/Initializer/random_uniform/subSubdrnn_classification/cell_0/bidirectional_rnn/bw/basic_lstm_cell/kernel/Initializer/random_uniform/maxdrnn_classification/cell_0/bidirectional_rnn/bw/basic_lstm_cell/kernel/Initializer/random_uniform/min*
T0*X
_classN
LJloc:@rnn_classification/cell_0/bidirectional_rnn/bw/basic_lstm_cell/kernel*
_output_shapes
: 
Ć
drnn_classification/cell_0/bidirectional_rnn/bw/basic_lstm_cell/kernel/Initializer/random_uniform/mulMulnrnn_classification/cell_0/bidirectional_rnn/bw/basic_lstm_cell/kernel/Initializer/random_uniform/RandomUniformdrnn_classification/cell_0/bidirectional_rnn/bw/basic_lstm_cell/kernel/Initializer/random_uniform/sub* 
_output_shapes
:
ŕ*
T0*X
_classN
LJloc:@rnn_classification/cell_0/bidirectional_rnn/bw/basic_lstm_cell/kernel
¸
`rnn_classification/cell_0/bidirectional_rnn/bw/basic_lstm_cell/kernel/Initializer/random_uniformAdddrnn_classification/cell_0/bidirectional_rnn/bw/basic_lstm_cell/kernel/Initializer/random_uniform/muldrnn_classification/cell_0/bidirectional_rnn/bw/basic_lstm_cell/kernel/Initializer/random_uniform/min*
T0*X
_classN
LJloc:@rnn_classification/cell_0/bidirectional_rnn/bw/basic_lstm_cell/kernel* 
_output_shapes
:
ŕ
ó
Ernn_classification/cell_0/bidirectional_rnn/bw/basic_lstm_cell/kernel
VariableV2*
dtype0* 
_output_shapes
:
ŕ*X
_classN
LJloc:@rnn_classification/cell_0/bidirectional_rnn/bw/basic_lstm_cell/kernel*
shape:
ŕ

Lrnn_classification/cell_0/bidirectional_rnn/bw/basic_lstm_cell/kernel/AssignAssignErnn_classification/cell_0/bidirectional_rnn/bw/basic_lstm_cell/kernel`rnn_classification/cell_0/bidirectional_rnn/bw/basic_lstm_cell/kernel/Initializer/random_uniform*
T0*X
_classN
LJloc:@rnn_classification/cell_0/bidirectional_rnn/bw/basic_lstm_cell/kernel* 
_output_shapes
:
ŕ
Č
Jrnn_classification/cell_0/bidirectional_rnn/bw/basic_lstm_cell/kernel/readIdentityErnn_classification/cell_0/bidirectional_rnn/bw/basic_lstm_cell/kernel*
T0* 
_output_shapes
:
ŕ
ü
Urnn_classification/cell_0/bidirectional_rnn/bw/basic_lstm_cell/bias/Initializer/zerosConst*
valueB*    *V
_classL
JHloc:@rnn_classification/cell_0/bidirectional_rnn/bw/basic_lstm_cell/bias*
dtype0*
_output_shapes	
:
ĺ
Crnn_classification/cell_0/bidirectional_rnn/bw/basic_lstm_cell/bias
VariableV2*
dtype0*
_output_shapes	
:*V
_classL
JHloc:@rnn_classification/cell_0/bidirectional_rnn/bw/basic_lstm_cell/bias*
shape:
î
Jrnn_classification/cell_0/bidirectional_rnn/bw/basic_lstm_cell/bias/AssignAssignCrnn_classification/cell_0/bidirectional_rnn/bw/basic_lstm_cell/biasUrnn_classification/cell_0/bidirectional_rnn/bw/basic_lstm_cell/bias/Initializer/zeros*
T0*V
_classL
JHloc:@rnn_classification/cell_0/bidirectional_rnn/bw/basic_lstm_cell/bias*
_output_shapes	
:
ż
Hrnn_classification/cell_0/bidirectional_rnn/bw/basic_lstm_cell/bias/readIdentityCrnn_classification/cell_0/bidirectional_rnn/bw/basic_lstm_cell/bias*
T0*
_output_shapes	
:
Ň
Mrnn_classification/cell_0/bidirectional_rnn/bw/bw/while/basic_lstm_cell/ConstConstA^rnn_classification/cell_0/bidirectional_rnn/bw/bw/while/Identity*
dtype0*
_output_shapes
: *
value	B :
Ř
Srnn_classification/cell_0/bidirectional_rnn/bw/bw/while/basic_lstm_cell/concat/axisConstA^rnn_classification/cell_0/bidirectional_rnn/bw/bw/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
ń
Nrnn_classification/cell_0/bidirectional_rnn/bw/bw/while/basic_lstm_cell/concatConcatV2Irnn_classification/cell_0/bidirectional_rnn/bw/bw/while/TensorArrayReadV3Brnn_classification/cell_0/bidirectional_rnn/bw/bw/while/Identity_4Srnn_classification/cell_0/bidirectional_rnn/bw/bw/while/basic_lstm_cell/concat/axis*
N*
_output_shapes
:	ŕ*
T0
¨
Nrnn_classification/cell_0/bidirectional_rnn/bw/bw/while/basic_lstm_cell/MatMulMatMulNrnn_classification/cell_0/bidirectional_rnn/bw/bw/while/basic_lstm_cell/concatTrnn_classification/cell_0/bidirectional_rnn/bw/bw/while/basic_lstm_cell/MatMul/Enter*
T0*
_output_shapes
:	
Ů
Trnn_classification/cell_0/bidirectional_rnn/bw/bw/while/basic_lstm_cell/MatMul/EnterEnterJrnn_classification/cell_0/bidirectional_rnn/bw/basic_lstm_cell/kernel/read*
T0*
is_constant(*
parallel_iterations *U

frame_nameGErnn_classification/cell_0/bidirectional_rnn/bw/bw/while/while_context* 
_output_shapes
:
ŕ
Ť
Ornn_classification/cell_0/bidirectional_rnn/bw/bw/while/basic_lstm_cell/BiasAddBiasAddNrnn_classification/cell_0/bidirectional_rnn/bw/bw/while/basic_lstm_cell/MatMulUrnn_classification/cell_0/bidirectional_rnn/bw/bw/while/basic_lstm_cell/BiasAdd/Enter*
T0*
_output_shapes
:	
Ó
Urnn_classification/cell_0/bidirectional_rnn/bw/bw/while/basic_lstm_cell/BiasAdd/EnterEnterHrnn_classification/cell_0/bidirectional_rnn/bw/basic_lstm_cell/bias/read*
T0*
is_constant(*
parallel_iterations *U

frame_nameGErnn_classification/cell_0/bidirectional_rnn/bw/bw/while/while_context*
_output_shapes	
:
Ô
Ornn_classification/cell_0/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Const_1ConstA^rnn_classification/cell_0/bidirectional_rnn/bw/bw/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
Ň
Mrnn_classification/cell_0/bidirectional_rnn/bw/bw/while/basic_lstm_cell/splitSplitMrnn_classification/cell_0/bidirectional_rnn/bw/bw/while/basic_lstm_cell/ConstOrnn_classification/cell_0/bidirectional_rnn/bw/bw/while/basic_lstm_cell/BiasAdd*
T0*
	num_split*@
_output_shapes.
,:	:	:	:	
×
Ornn_classification/cell_0/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Const_2ConstA^rnn_classification/cell_0/bidirectional_rnn/bw/bw/while/Identity*
valueB
 *  ?*
dtype0*
_output_shapes
: 

Krnn_classification/cell_0/bidirectional_rnn/bw/bw/while/basic_lstm_cell/AddAddOrnn_classification/cell_0/bidirectional_rnn/bw/bw/while/basic_lstm_cell/split:2Ornn_classification/cell_0/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Const_2*
T0*
_output_shapes
:	
Ń
Ornn_classification/cell_0/bidirectional_rnn/bw/bw/while/basic_lstm_cell/SigmoidSigmoidKrnn_classification/cell_0/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Add*
T0*
_output_shapes
:	

Krnn_classification/cell_0/bidirectional_rnn/bw/bw/while/basic_lstm_cell/MulMulBrnn_classification/cell_0/bidirectional_rnn/bw/bw/while/Identity_3Ornn_classification/cell_0/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Sigmoid*
_output_shapes
:	*
T0
Ő
Qrnn_classification/cell_0/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Sigmoid_1SigmoidMrnn_classification/cell_0/bidirectional_rnn/bw/bw/while/basic_lstm_cell/split*
T0*
_output_shapes
:	
Ď
Lrnn_classification/cell_0/bidirectional_rnn/bw/bw/while/basic_lstm_cell/TanhTanhOrnn_classification/cell_0/bidirectional_rnn/bw/bw/while/basic_lstm_cell/split:1*
T0*
_output_shapes
:	

Mrnn_classification/cell_0/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_1MulQrnn_classification/cell_0/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Sigmoid_1Lrnn_classification/cell_0/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Tanh*
T0*
_output_shapes
:	

Mrnn_classification/cell_0/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Add_1AddKrnn_classification/cell_0/bidirectional_rnn/bw/bw/while/basic_lstm_cell/MulMrnn_classification/cell_0/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_1*
T0*
_output_shapes
:	
Ď
Nrnn_classification/cell_0/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Tanh_1TanhMrnn_classification/cell_0/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Add_1*
T0*
_output_shapes
:	
×
Qrnn_classification/cell_0/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Sigmoid_2SigmoidOrnn_classification/cell_0/bidirectional_rnn/bw/bw/while/basic_lstm_cell/split:3*
T0*
_output_shapes
:	
Ą
Mrnn_classification/cell_0/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_2MulNrnn_classification/cell_0/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Tanh_1Qrnn_classification/cell_0/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Sigmoid_2*
T0*
_output_shapes
:	
Ż
>rnn_classification/cell_0/bidirectional_rnn/bw/bw/while/SelectSelectDrnn_classification/cell_0/bidirectional_rnn/bw/bw/while/GreaterEqualDrnn_classification/cell_0/bidirectional_rnn/bw/bw/while/Select/EnterMrnn_classification/cell_0/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_2*
T0*`
_classV
TRloc:@rnn_classification/cell_0/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_2*
_output_shapes
:	

Drnn_classification/cell_0/bidirectional_rnn/bw/bw/while/Select/EnterEnter7rnn_classification/cell_0/bidirectional_rnn/bw/bw/zeros*
T0*`
_classV
TRloc:@rnn_classification/cell_0/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_2*
parallel_iterations *
is_constant(*U

frame_nameGErnn_classification/cell_0/bidirectional_rnn/bw/bw/while/while_context*
_output_shapes
:	
Ż
@rnn_classification/cell_0/bidirectional_rnn/bw/bw/while/Select_1SelectDrnn_classification/cell_0/bidirectional_rnn/bw/bw/while/GreaterEqualBrnn_classification/cell_0/bidirectional_rnn/bw/bw/while/Identity_3Mrnn_classification/cell_0/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Add_1*
T0*`
_classV
TRloc:@rnn_classification/cell_0/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Add_1*
_output_shapes
:	
Ż
@rnn_classification/cell_0/bidirectional_rnn/bw/bw/while/Select_2SelectDrnn_classification/cell_0/bidirectional_rnn/bw/bw/while/GreaterEqualBrnn_classification/cell_0/bidirectional_rnn/bw/bw/while/Identity_4Mrnn_classification/cell_0/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_2*
T0*`
_classV
TRloc:@rnn_classification/cell_0/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_2*
_output_shapes
:	

[rnn_classification/cell_0/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3TensorArrayWriteV3arnn_classification/cell_0/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3/EnterBrnn_classification/cell_0/bidirectional_rnn/bw/bw/while/Identity_1>rnn_classification/cell_0/bidirectional_rnn/bw/bw/while/SelectBrnn_classification/cell_0/bidirectional_rnn/bw/bw/while/Identity_2*
T0*`
_classV
TRloc:@rnn_classification/cell_0/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_2*
_output_shapes
: 
ľ
arnn_classification/cell_0/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3/EnterEnter=rnn_classification/cell_0/bidirectional_rnn/bw/bw/TensorArray*
is_constant(*U

frame_nameGErnn_classification/cell_0/bidirectional_rnn/bw/bw/while/while_context*
_output_shapes
:*
T0*`
_classV
TRloc:@rnn_classification/cell_0/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_2*
parallel_iterations 
Ä
?rnn_classification/cell_0/bidirectional_rnn/bw/bw/while/add_1/yConstA^rnn_classification/cell_0/bidirectional_rnn/bw/bw/while/Identity*
dtype0*
_output_shapes
: *
value	B :
ę
=rnn_classification/cell_0/bidirectional_rnn/bw/bw/while/add_1AddBrnn_classification/cell_0/bidirectional_rnn/bw/bw/while/Identity_1?rnn_classification/cell_0/bidirectional_rnn/bw/bw/while/add_1/y*
T0*
_output_shapes
: 
´
Ernn_classification/cell_0/bidirectional_rnn/bw/bw/while/NextIterationNextIteration;rnn_classification/cell_0/bidirectional_rnn/bw/bw/while/add*
T0*
_output_shapes
: 
¸
Grnn_classification/cell_0/bidirectional_rnn/bw/bw/while/NextIteration_1NextIteration=rnn_classification/cell_0/bidirectional_rnn/bw/bw/while/add_1*
T0*
_output_shapes
: 
Ö
Grnn_classification/cell_0/bidirectional_rnn/bw/bw/while/NextIteration_2NextIteration[rnn_classification/cell_0/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3*
T0*
_output_shapes
: 
Ä
Grnn_classification/cell_0/bidirectional_rnn/bw/bw/while/NextIteration_3NextIteration@rnn_classification/cell_0/bidirectional_rnn/bw/bw/while/Select_1*
T0*
_output_shapes
:	
Ä
Grnn_classification/cell_0/bidirectional_rnn/bw/bw/while/NextIteration_4NextIteration@rnn_classification/cell_0/bidirectional_rnn/bw/bw/while/Select_2*
T0*
_output_shapes
:	
Ľ
<rnn_classification/cell_0/bidirectional_rnn/bw/bw/while/ExitExit>rnn_classification/cell_0/bidirectional_rnn/bw/bw/while/Switch*
_output_shapes
: *
T0
Š
>rnn_classification/cell_0/bidirectional_rnn/bw/bw/while/Exit_1Exit@rnn_classification/cell_0/bidirectional_rnn/bw/bw/while/Switch_1*
T0*
_output_shapes
: 
Š
>rnn_classification/cell_0/bidirectional_rnn/bw/bw/while/Exit_2Exit@rnn_classification/cell_0/bidirectional_rnn/bw/bw/while/Switch_2*
T0*
_output_shapes
: 
˛
>rnn_classification/cell_0/bidirectional_rnn/bw/bw/while/Exit_3Exit@rnn_classification/cell_0/bidirectional_rnn/bw/bw/while/Switch_3*
T0*
_output_shapes
:	
˛
>rnn_classification/cell_0/bidirectional_rnn/bw/bw/while/Exit_4Exit@rnn_classification/cell_0/bidirectional_rnn/bw/bw/while/Switch_4*
_output_shapes
:	*
T0
Ň
Trnn_classification/cell_0/bidirectional_rnn/bw/bw/TensorArrayStack/TensorArraySizeV3TensorArraySizeV3=rnn_classification/cell_0/bidirectional_rnn/bw/bw/TensorArray>rnn_classification/cell_0/bidirectional_rnn/bw/bw/while/Exit_2*P
_classF
DBloc:@rnn_classification/cell_0/bidirectional_rnn/bw/bw/TensorArray*
_output_shapes
: 
â
Nrnn_classification/cell_0/bidirectional_rnn/bw/bw/TensorArrayStack/range/startConst*
value	B : *P
_classF
DBloc:@rnn_classification/cell_0/bidirectional_rnn/bw/bw/TensorArray*
dtype0*
_output_shapes
: 
â
Nrnn_classification/cell_0/bidirectional_rnn/bw/bw/TensorArrayStack/range/deltaConst*
value	B :*P
_classF
DBloc:@rnn_classification/cell_0/bidirectional_rnn/bw/bw/TensorArray*
dtype0*
_output_shapes
: 
ž
Hrnn_classification/cell_0/bidirectional_rnn/bw/bw/TensorArrayStack/rangeRangeNrnn_classification/cell_0/bidirectional_rnn/bw/bw/TensorArrayStack/range/startTrnn_classification/cell_0/bidirectional_rnn/bw/bw/TensorArrayStack/TensorArraySizeV3Nrnn_classification/cell_0/bidirectional_rnn/bw/bw/TensorArrayStack/range/delta*P
_classF
DBloc:@rnn_classification/cell_0/bidirectional_rnn/bw/bw/TensorArray*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
á
Vrnn_classification/cell_0/bidirectional_rnn/bw/bw/TensorArrayStack/TensorArrayGatherV3TensorArrayGatherV3=rnn_classification/cell_0/bidirectional_rnn/bw/bw/TensorArrayHrnn_classification/cell_0/bidirectional_rnn/bw/bw/TensorArrayStack/range>rnn_classification/cell_0/bidirectional_rnn/bw/bw/while/Exit_2*
element_shape:	*P
_classF
DBloc:@rnn_classification/cell_0/bidirectional_rnn/bw/bw/TensorArray*
dtype0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙

9rnn_classification/cell_0/bidirectional_rnn/bw/bw/Const_5Const*
valueB:*
dtype0*
_output_shapes
:
z
8rnn_classification/cell_0/bidirectional_rnn/bw/bw/Rank_1Const*
value	B :*
dtype0*
_output_shapes
: 

?rnn_classification/cell_0/bidirectional_rnn/bw/bw/range_1/startConst*
dtype0*
_output_shapes
: *
value	B :

?rnn_classification/cell_0/bidirectional_rnn/bw/bw/range_1/deltaConst*
value	B :*
dtype0*
_output_shapes
: 

9rnn_classification/cell_0/bidirectional_rnn/bw/bw/range_1Range?rnn_classification/cell_0/bidirectional_rnn/bw/bw/range_1/start8rnn_classification/cell_0/bidirectional_rnn/bw/bw/Rank_1?rnn_classification/cell_0/bidirectional_rnn/bw/bw/range_1/delta*
_output_shapes
:

Crnn_classification/cell_0/bidirectional_rnn/bw/bw/concat_2/values_0Const*
valueB"       *
dtype0*
_output_shapes
:

?rnn_classification/cell_0/bidirectional_rnn/bw/bw/concat_2/axisConst*
value	B : *
dtype0*
_output_shapes
: 
ľ
:rnn_classification/cell_0/bidirectional_rnn/bw/bw/concat_2ConcatV2Crnn_classification/cell_0/bidirectional_rnn/bw/bw/concat_2/values_09rnn_classification/cell_0/bidirectional_rnn/bw/bw/range_1?rnn_classification/cell_0/bidirectional_rnn/bw/bw/concat_2/axis*
N*
_output_shapes
:*
T0

=rnn_classification/cell_0/bidirectional_rnn/bw/bw/transpose_1	TransposeVrnn_classification/cell_0/bidirectional_rnn/bw/bw/TensorArrayStack/TensorArrayGatherV3:rnn_classification/cell_0/bidirectional_rnn/bw/bw/concat_2*,
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ę
)rnn_classification/cell_0/ReverseSequenceReverseSequence=rnn_classification/cell_0/bidirectional_rnn/bw/bw/transpose_1Squeeze*
T0*
seq_dim*,
_output_shapes
:˙˙˙˙˙˙˙˙˙
g
%rnn_classification/cell_0/concat/axisConst*
value	B :*
dtype0*
_output_shapes
: 
ý
 rnn_classification/cell_0/concatConcatV2=rnn_classification/cell_0/bidirectional_rnn/fw/fw/transpose_1)rnn_classification/cell_0/ReverseSequence%rnn_classification/cell_0/concat/axis*
T0*
N*,
_output_shapes
:˙˙˙˙˙˙˙˙˙
x
6rnn_classification/cell_1/bidirectional_rnn/fw/fw/RankConst*
value	B :*
dtype0*
_output_shapes
: 

=rnn_classification/cell_1/bidirectional_rnn/fw/fw/range/startConst*
value	B :*
dtype0*
_output_shapes
: 

=rnn_classification/cell_1/bidirectional_rnn/fw/fw/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 

7rnn_classification/cell_1/bidirectional_rnn/fw/fw/rangeRange=rnn_classification/cell_1/bidirectional_rnn/fw/fw/range/start6rnn_classification/cell_1/bidirectional_rnn/fw/fw/Rank=rnn_classification/cell_1/bidirectional_rnn/fw/fw/range/delta*
_output_shapes
:

Arnn_classification/cell_1/bidirectional_rnn/fw/fw/concat/values_0Const*
valueB"       *
dtype0*
_output_shapes
:

=rnn_classification/cell_1/bidirectional_rnn/fw/fw/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
­
8rnn_classification/cell_1/bidirectional_rnn/fw/fw/concatConcatV2Arnn_classification/cell_1/bidirectional_rnn/fw/fw/concat/values_07rnn_classification/cell_1/bidirectional_rnn/fw/fw/range=rnn_classification/cell_1/bidirectional_rnn/fw/fw/concat/axis*
T0*
N*
_output_shapes
:
Ű
;rnn_classification/cell_1/bidirectional_rnn/fw/fw/transpose	Transpose rnn_classification/cell_0/concat8rnn_classification/cell_1/bidirectional_rnn/fw/fw/concat*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙
~
9rnn_classification/cell_1/bidirectional_rnn/fw/fw/ToInt32CastSqueeze*

SrcT0	*

DstT0*
_output_shapes
:
­
Arnn_classification/cell_1/bidirectional_rnn/fw/fw/sequence_lengthIdentity9rnn_classification/cell_1/bidirectional_rnn/fw/fw/ToInt32*
T0*
_output_shapes
:
°
frnn_classification/cell_1/bidirectional_rnn/fw/fw/DropoutWrapperZeroState/BasicLSTMCellZeroState/ConstConst*
valueB:*
dtype0*
_output_shapes
:
ł
hrnn_classification/cell_1/bidirectional_rnn/fw/fw/DropoutWrapperZeroState/BasicLSTMCellZeroState/Const_1Const*
dtype0*
_output_shapes
:*
valueB:
Ž
lrnn_classification/cell_1/bidirectional_rnn/fw/fw/DropoutWrapperZeroState/BasicLSTMCellZeroState/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
á
grnn_classification/cell_1/bidirectional_rnn/fw/fw/DropoutWrapperZeroState/BasicLSTMCellZeroState/concatConcatV2frnn_classification/cell_1/bidirectional_rnn/fw/fw/DropoutWrapperZeroState/BasicLSTMCellZeroState/Consthrnn_classification/cell_1/bidirectional_rnn/fw/fw/DropoutWrapperZeroState/BasicLSTMCellZeroState/Const_1lrnn_classification/cell_1/bidirectional_rnn/fw/fw/DropoutWrapperZeroState/BasicLSTMCellZeroState/concat/axis*
T0*
N*
_output_shapes
:
ą
lrnn_classification/cell_1/bidirectional_rnn/fw/fw/DropoutWrapperZeroState/BasicLSTMCellZeroState/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
ď
frnn_classification/cell_1/bidirectional_rnn/fw/fw/DropoutWrapperZeroState/BasicLSTMCellZeroState/zerosFillgrnn_classification/cell_1/bidirectional_rnn/fw/fw/DropoutWrapperZeroState/BasicLSTMCellZeroState/concatlrnn_classification/cell_1/bidirectional_rnn/fw/fw/DropoutWrapperZeroState/BasicLSTMCellZeroState/zeros/Const*
T0*
_output_shapes
:	
˛
hrnn_classification/cell_1/bidirectional_rnn/fw/fw/DropoutWrapperZeroState/BasicLSTMCellZeroState/Const_2Const*
valueB:*
dtype0*
_output_shapes
:
ł
hrnn_classification/cell_1/bidirectional_rnn/fw/fw/DropoutWrapperZeroState/BasicLSTMCellZeroState/Const_3Const*
valueB:*
dtype0*
_output_shapes
:
˛
hrnn_classification/cell_1/bidirectional_rnn/fw/fw/DropoutWrapperZeroState/BasicLSTMCellZeroState/Const_4Const*
valueB:*
dtype0*
_output_shapes
:
ł
hrnn_classification/cell_1/bidirectional_rnn/fw/fw/DropoutWrapperZeroState/BasicLSTMCellZeroState/Const_5Const*
dtype0*
_output_shapes
:*
valueB:
°
nrnn_classification/cell_1/bidirectional_rnn/fw/fw/DropoutWrapperZeroState/BasicLSTMCellZeroState/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
ç
irnn_classification/cell_1/bidirectional_rnn/fw/fw/DropoutWrapperZeroState/BasicLSTMCellZeroState/concat_1ConcatV2hrnn_classification/cell_1/bidirectional_rnn/fw/fw/DropoutWrapperZeroState/BasicLSTMCellZeroState/Const_4hrnn_classification/cell_1/bidirectional_rnn/fw/fw/DropoutWrapperZeroState/BasicLSTMCellZeroState/Const_5nrnn_classification/cell_1/bidirectional_rnn/fw/fw/DropoutWrapperZeroState/BasicLSTMCellZeroState/concat_1/axis*
N*
_output_shapes
:*
T0
ł
nrnn_classification/cell_1/bidirectional_rnn/fw/fw/DropoutWrapperZeroState/BasicLSTMCellZeroState/zeros_1/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
ő
hrnn_classification/cell_1/bidirectional_rnn/fw/fw/DropoutWrapperZeroState/BasicLSTMCellZeroState/zeros_1Fillirnn_classification/cell_1/bidirectional_rnn/fw/fw/DropoutWrapperZeroState/BasicLSTMCellZeroState/concat_1nrnn_classification/cell_1/bidirectional_rnn/fw/fw/DropoutWrapperZeroState/BasicLSTMCellZeroState/zeros_1/Const*
T0*
_output_shapes
:	
˛
hrnn_classification/cell_1/bidirectional_rnn/fw/fw/DropoutWrapperZeroState/BasicLSTMCellZeroState/Const_6Const*
valueB:*
dtype0*
_output_shapes
:
ł
hrnn_classification/cell_1/bidirectional_rnn/fw/fw/DropoutWrapperZeroState/BasicLSTMCellZeroState/Const_7Const*
valueB:*
dtype0*
_output_shapes
:

7rnn_classification/cell_1/bidirectional_rnn/fw/fw/ShapeConst*
valueB:*
dtype0*
_output_shapes
:

7rnn_classification/cell_1/bidirectional_rnn/fw/fw/stackConst*
valueB:*
dtype0*
_output_shapes
:
×
7rnn_classification/cell_1/bidirectional_rnn/fw/fw/EqualEqual7rnn_classification/cell_1/bidirectional_rnn/fw/fw/Shape7rnn_classification/cell_1/bidirectional_rnn/fw/fw/stack*
_output_shapes
:*
T0

7rnn_classification/cell_1/bidirectional_rnn/fw/fw/ConstConst*
valueB: *
dtype0*
_output_shapes
:
Ć
5rnn_classification/cell_1/bidirectional_rnn/fw/fw/AllAll7rnn_classification/cell_1/bidirectional_rnn/fw/fw/Equal7rnn_classification/cell_1/bidirectional_rnn/fw/fw/Const*
_output_shapes
: 
ŕ
>rnn_classification/cell_1/bidirectional_rnn/fw/fw/Assert/ConstConst*r
valueiBg BaExpected shape for Tensor rnn_classification/cell_1/bidirectional_rnn/fw/fw/sequence_length:0 is *
dtype0*
_output_shapes
: 

@rnn_classification/cell_1/bidirectional_rnn/fw/fw/Assert/Const_1Const*!
valueB B but saw shape: *
dtype0*
_output_shapes
: 
č
Frnn_classification/cell_1/bidirectional_rnn/fw/fw/Assert/Assert/data_0Const*r
valueiBg BaExpected shape for Tensor rnn_classification/cell_1/bidirectional_rnn/fw/fw/sequence_length:0 is *
dtype0*
_output_shapes
: 

Frnn_classification/cell_1/bidirectional_rnn/fw/fw/Assert/Assert/data_2Const*!
valueB B but saw shape: *
dtype0*
_output_shapes
: 

?rnn_classification/cell_1/bidirectional_rnn/fw/fw/Assert/AssertAssert5rnn_classification/cell_1/bidirectional_rnn/fw/fw/AllFrnn_classification/cell_1/bidirectional_rnn/fw/fw/Assert/Assert/data_07rnn_classification/cell_1/bidirectional_rnn/fw/fw/stackFrnn_classification/cell_1/bidirectional_rnn/fw/fw/Assert/Assert/data_27rnn_classification/cell_1/bidirectional_rnn/fw/fw/Shape*
T
2
ó
=rnn_classification/cell_1/bidirectional_rnn/fw/fw/CheckSeqLenIdentityArnn_classification/cell_1/bidirectional_rnn/fw/fw/sequence_length@^rnn_classification/cell_1/bidirectional_rnn/fw/fw/Assert/Assert*
T0*
_output_shapes
:
¤
9rnn_classification/cell_1/bidirectional_rnn/fw/fw/Shape_1Shape;rnn_classification/cell_1/bidirectional_rnn/fw/fw/transpose*
T0*
_output_shapes
:

Ernn_classification/cell_1/bidirectional_rnn/fw/fw/strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB: 

Grnn_classification/cell_1/bidirectional_rnn/fw/fw/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB:

Grnn_classification/cell_1/bidirectional_rnn/fw/fw/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
Š
?rnn_classification/cell_1/bidirectional_rnn/fw/fw/strided_sliceStridedSlice9rnn_classification/cell_1/bidirectional_rnn/fw/fw/Shape_1Ernn_classification/cell_1/bidirectional_rnn/fw/fw/strided_slice/stackGrnn_classification/cell_1/bidirectional_rnn/fw/fw/strided_slice/stack_1Grnn_classification/cell_1/bidirectional_rnn/fw/fw/strided_slice/stack_2*
_output_shapes
: *
T0*
Index0*
shrink_axis_mask

9rnn_classification/cell_1/bidirectional_rnn/fw/fw/Const_1Const*
valueB:*
dtype0*
_output_shapes
:

9rnn_classification/cell_1/bidirectional_rnn/fw/fw/Const_2Const*
valueB:*
dtype0*
_output_shapes
:

?rnn_classification/cell_1/bidirectional_rnn/fw/fw/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
Ť
:rnn_classification/cell_1/bidirectional_rnn/fw/fw/concat_1ConcatV29rnn_classification/cell_1/bidirectional_rnn/fw/fw/Const_19rnn_classification/cell_1/bidirectional_rnn/fw/fw/Const_2?rnn_classification/cell_1/bidirectional_rnn/fw/fw/concat_1/axis*
T0*
N*
_output_shapes
:

=rnn_classification/cell_1/bidirectional_rnn/fw/fw/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
ä
7rnn_classification/cell_1/bidirectional_rnn/fw/fw/zerosFill:rnn_classification/cell_1/bidirectional_rnn/fw/fw/concat_1=rnn_classification/cell_1/bidirectional_rnn/fw/fw/zeros/Const*
T0*
_output_shapes
:	

9rnn_classification/cell_1/bidirectional_rnn/fw/fw/Const_3Const*
valueB: *
dtype0*
_output_shapes
:
×
5rnn_classification/cell_1/bidirectional_rnn/fw/fw/MinMin=rnn_classification/cell_1/bidirectional_rnn/fw/fw/CheckSeqLen9rnn_classification/cell_1/bidirectional_rnn/fw/fw/Const_3*
_output_shapes
: *
T0

9rnn_classification/cell_1/bidirectional_rnn/fw/fw/Const_4Const*
valueB: *
dtype0*
_output_shapes
:
×
5rnn_classification/cell_1/bidirectional_rnn/fw/fw/MaxMax=rnn_classification/cell_1/bidirectional_rnn/fw/fw/CheckSeqLen9rnn_classification/cell_1/bidirectional_rnn/fw/fw/Const_4*
T0*
_output_shapes
: 
x
6rnn_classification/cell_1/bidirectional_rnn/fw/fw/timeConst*
dtype0*
_output_shapes
: *
value	B : 
×
=rnn_classification/cell_1/bidirectional_rnn/fw/fw/TensorArrayTensorArrayV3?rnn_classification/cell_1/bidirectional_rnn/fw/fw/strided_slice*]
tensor_array_nameHFrnn_classification/cell_1/bidirectional_rnn/fw/fw/dynamic_rnn/output_0*
dtype0*
_output_shapes

:: *
element_shape:	*
identical_element_shapes(
Ř
?rnn_classification/cell_1/bidirectional_rnn/fw/fw/TensorArray_1TensorArrayV3?rnn_classification/cell_1/bidirectional_rnn/fw/fw/strided_slice*
identical_element_shapes(*\
tensor_array_nameGErnn_classification/cell_1/bidirectional_rnn/fw/fw/dynamic_rnn/input_0*
dtype0*
_output_shapes

:: *
element_shape:	
ľ
Jrnn_classification/cell_1/bidirectional_rnn/fw/fw/TensorArrayUnstack/ShapeShape;rnn_classification/cell_1/bidirectional_rnn/fw/fw/transpose*
T0*
_output_shapes
:
˘
Xrnn_classification/cell_1/bidirectional_rnn/fw/fw/TensorArrayUnstack/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
¤
Zrnn_classification/cell_1/bidirectional_rnn/fw/fw/TensorArrayUnstack/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
¤
Zrnn_classification/cell_1/bidirectional_rnn/fw/fw/TensorArrayUnstack/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:

Rrnn_classification/cell_1/bidirectional_rnn/fw/fw/TensorArrayUnstack/strided_sliceStridedSliceJrnn_classification/cell_1/bidirectional_rnn/fw/fw/TensorArrayUnstack/ShapeXrnn_classification/cell_1/bidirectional_rnn/fw/fw/TensorArrayUnstack/strided_slice/stackZrnn_classification/cell_1/bidirectional_rnn/fw/fw/TensorArrayUnstack/strided_slice/stack_1Zrnn_classification/cell_1/bidirectional_rnn/fw/fw/TensorArrayUnstack/strided_slice/stack_2*
Index0*
T0*
shrink_axis_mask*
_output_shapes
: 

Prnn_classification/cell_1/bidirectional_rnn/fw/fw/TensorArrayUnstack/range/startConst*
dtype0*
_output_shapes
: *
value	B : 

Prnn_classification/cell_1/bidirectional_rnn/fw/fw/TensorArrayUnstack/range/deltaConst*
dtype0*
_output_shapes
: *
value	B :
đ
Jrnn_classification/cell_1/bidirectional_rnn/fw/fw/TensorArrayUnstack/rangeRangePrnn_classification/cell_1/bidirectional_rnn/fw/fw/TensorArrayUnstack/range/startRrnn_classification/cell_1/bidirectional_rnn/fw/fw/TensorArrayUnstack/strided_slicePrnn_classification/cell_1/bidirectional_rnn/fw/fw/TensorArrayUnstack/range/delta*#
_output_shapes
:˙˙˙˙˙˙˙˙˙

lrnn_classification/cell_1/bidirectional_rnn/fw/fw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3TensorArrayScatterV3?rnn_classification/cell_1/bidirectional_rnn/fw/fw/TensorArray_1Jrnn_classification/cell_1/bidirectional_rnn/fw/fw/TensorArrayUnstack/range;rnn_classification/cell_1/bidirectional_rnn/fw/fw/transposeArnn_classification/cell_1/bidirectional_rnn/fw/fw/TensorArray_1:1*
_output_shapes
: *
T0*N
_classD
B@loc:@rnn_classification/cell_1/bidirectional_rnn/fw/fw/transpose
}
;rnn_classification/cell_1/bidirectional_rnn/fw/fw/Maximum/xConst*
dtype0*
_output_shapes
: *
value	B :
Ů
9rnn_classification/cell_1/bidirectional_rnn/fw/fw/MaximumMaximum;rnn_classification/cell_1/bidirectional_rnn/fw/fw/Maximum/x5rnn_classification/cell_1/bidirectional_rnn/fw/fw/Max*
_output_shapes
: *
T0
á
9rnn_classification/cell_1/bidirectional_rnn/fw/fw/MinimumMinimum?rnn_classification/cell_1/bidirectional_rnn/fw/fw/strided_slice9rnn_classification/cell_1/bidirectional_rnn/fw/fw/Maximum*
T0*
_output_shapes
: 

Irnn_classification/cell_1/bidirectional_rnn/fw/fw/while/iteration_counterConst*
value	B : *
dtype0*
_output_shapes
: 
¤
=rnn_classification/cell_1/bidirectional_rnn/fw/fw/while/EnterEnterIrnn_classification/cell_1/bidirectional_rnn/fw/fw/while/iteration_counter*
T0*
parallel_iterations *U

frame_nameGErnn_classification/cell_1/bidirectional_rnn/fw/fw/while/while_context*
_output_shapes
: 

?rnn_classification/cell_1/bidirectional_rnn/fw/fw/while/Enter_1Enter6rnn_classification/cell_1/bidirectional_rnn/fw/fw/time*
T0*
parallel_iterations *U

frame_nameGErnn_classification/cell_1/bidirectional_rnn/fw/fw/while/while_context*
_output_shapes
: 

?rnn_classification/cell_1/bidirectional_rnn/fw/fw/while/Enter_2Enter?rnn_classification/cell_1/bidirectional_rnn/fw/fw/TensorArray:1*
T0*
parallel_iterations *U

frame_nameGErnn_classification/cell_1/bidirectional_rnn/fw/fw/while/while_context*
_output_shapes
: 
Ě
?rnn_classification/cell_1/bidirectional_rnn/fw/fw/while/Enter_3Enterfrnn_classification/cell_1/bidirectional_rnn/fw/fw/DropoutWrapperZeroState/BasicLSTMCellZeroState/zeros*
parallel_iterations *U

frame_nameGErnn_classification/cell_1/bidirectional_rnn/fw/fw/while/while_context*
_output_shapes
:	*
T0
Î
?rnn_classification/cell_1/bidirectional_rnn/fw/fw/while/Enter_4Enterhrnn_classification/cell_1/bidirectional_rnn/fw/fw/DropoutWrapperZeroState/BasicLSTMCellZeroState/zeros_1*
parallel_iterations *U

frame_nameGErnn_classification/cell_1/bidirectional_rnn/fw/fw/while/while_context*
_output_shapes
:	*
T0
ř
=rnn_classification/cell_1/bidirectional_rnn/fw/fw/while/MergeMerge=rnn_classification/cell_1/bidirectional_rnn/fw/fw/while/EnterErnn_classification/cell_1/bidirectional_rnn/fw/fw/while/NextIteration*
N*
_output_shapes
: : *
T0
ţ
?rnn_classification/cell_1/bidirectional_rnn/fw/fw/while/Merge_1Merge?rnn_classification/cell_1/bidirectional_rnn/fw/fw/while/Enter_1Grnn_classification/cell_1/bidirectional_rnn/fw/fw/while/NextIteration_1*
T0*
N*
_output_shapes
: : 
ţ
?rnn_classification/cell_1/bidirectional_rnn/fw/fw/while/Merge_2Merge?rnn_classification/cell_1/bidirectional_rnn/fw/fw/while/Enter_2Grnn_classification/cell_1/bidirectional_rnn/fw/fw/while/NextIteration_2*
T0*
N*
_output_shapes
: : 

?rnn_classification/cell_1/bidirectional_rnn/fw/fw/while/Merge_3Merge?rnn_classification/cell_1/bidirectional_rnn/fw/fw/while/Enter_3Grnn_classification/cell_1/bidirectional_rnn/fw/fw/while/NextIteration_3*
T0*
N*!
_output_shapes
:	: 

?rnn_classification/cell_1/bidirectional_rnn/fw/fw/while/Merge_4Merge?rnn_classification/cell_1/bidirectional_rnn/fw/fw/while/Enter_4Grnn_classification/cell_1/bidirectional_rnn/fw/fw/while/NextIteration_4*
T0*
N*!
_output_shapes
:	: 
č
<rnn_classification/cell_1/bidirectional_rnn/fw/fw/while/LessLess=rnn_classification/cell_1/bidirectional_rnn/fw/fw/while/MergeBrnn_classification/cell_1/bidirectional_rnn/fw/fw/while/Less/Enter*
T0*
_output_shapes
: 
˛
Brnn_classification/cell_1/bidirectional_rnn/fw/fw/while/Less/EnterEnter?rnn_classification/cell_1/bidirectional_rnn/fw/fw/strided_slice*
T0*
is_constant(*
parallel_iterations *U

frame_nameGErnn_classification/cell_1/bidirectional_rnn/fw/fw/while/while_context*
_output_shapes
: 
î
>rnn_classification/cell_1/bidirectional_rnn/fw/fw/while/Less_1Less?rnn_classification/cell_1/bidirectional_rnn/fw/fw/while/Merge_1Drnn_classification/cell_1/bidirectional_rnn/fw/fw/while/Less_1/Enter*
T0*
_output_shapes
: 
Ž
Drnn_classification/cell_1/bidirectional_rnn/fw/fw/while/Less_1/EnterEnter9rnn_classification/cell_1/bidirectional_rnn/fw/fw/Minimum*
T0*
is_constant(*
parallel_iterations *U

frame_nameGErnn_classification/cell_1/bidirectional_rnn/fw/fw/while/while_context*
_output_shapes
: 
ć
Brnn_classification/cell_1/bidirectional_rnn/fw/fw/while/LogicalAnd
LogicalAnd<rnn_classification/cell_1/bidirectional_rnn/fw/fw/while/Less>rnn_classification/cell_1/bidirectional_rnn/fw/fw/while/Less_1*
_output_shapes
: 
¨
@rnn_classification/cell_1/bidirectional_rnn/fw/fw/while/LoopCondLoopCondBrnn_classification/cell_1/bidirectional_rnn/fw/fw/while/LogicalAnd*
_output_shapes
: 
ž
>rnn_classification/cell_1/bidirectional_rnn/fw/fw/while/SwitchSwitch=rnn_classification/cell_1/bidirectional_rnn/fw/fw/while/Merge@rnn_classification/cell_1/bidirectional_rnn/fw/fw/while/LoopCond*
T0*P
_classF
DBloc:@rnn_classification/cell_1/bidirectional_rnn/fw/fw/while/Merge*
_output_shapes
: : 
Ä
@rnn_classification/cell_1/bidirectional_rnn/fw/fw/while/Switch_1Switch?rnn_classification/cell_1/bidirectional_rnn/fw/fw/while/Merge_1@rnn_classification/cell_1/bidirectional_rnn/fw/fw/while/LoopCond*
T0*R
_classH
FDloc:@rnn_classification/cell_1/bidirectional_rnn/fw/fw/while/Merge_1*
_output_shapes
: : 
Ä
@rnn_classification/cell_1/bidirectional_rnn/fw/fw/while/Switch_2Switch?rnn_classification/cell_1/bidirectional_rnn/fw/fw/while/Merge_2@rnn_classification/cell_1/bidirectional_rnn/fw/fw/while/LoopCond*
T0*R
_classH
FDloc:@rnn_classification/cell_1/bidirectional_rnn/fw/fw/while/Merge_2*
_output_shapes
: : 
Ö
@rnn_classification/cell_1/bidirectional_rnn/fw/fw/while/Switch_3Switch?rnn_classification/cell_1/bidirectional_rnn/fw/fw/while/Merge_3@rnn_classification/cell_1/bidirectional_rnn/fw/fw/while/LoopCond*
T0*R
_classH
FDloc:@rnn_classification/cell_1/bidirectional_rnn/fw/fw/while/Merge_3**
_output_shapes
:	:	
Ö
@rnn_classification/cell_1/bidirectional_rnn/fw/fw/while/Switch_4Switch?rnn_classification/cell_1/bidirectional_rnn/fw/fw/while/Merge_4@rnn_classification/cell_1/bidirectional_rnn/fw/fw/while/LoopCond**
_output_shapes
:	:	*
T0*R
_classH
FDloc:@rnn_classification/cell_1/bidirectional_rnn/fw/fw/while/Merge_4
Ż
@rnn_classification/cell_1/bidirectional_rnn/fw/fw/while/IdentityIdentity@rnn_classification/cell_1/bidirectional_rnn/fw/fw/while/Switch:1*
T0*
_output_shapes
: 
ł
Brnn_classification/cell_1/bidirectional_rnn/fw/fw/while/Identity_1IdentityBrnn_classification/cell_1/bidirectional_rnn/fw/fw/while/Switch_1:1*
T0*
_output_shapes
: 
ł
Brnn_classification/cell_1/bidirectional_rnn/fw/fw/while/Identity_2IdentityBrnn_classification/cell_1/bidirectional_rnn/fw/fw/while/Switch_2:1*
T0*
_output_shapes
: 
ź
Brnn_classification/cell_1/bidirectional_rnn/fw/fw/while/Identity_3IdentityBrnn_classification/cell_1/bidirectional_rnn/fw/fw/while/Switch_3:1*
T0*
_output_shapes
:	
ź
Brnn_classification/cell_1/bidirectional_rnn/fw/fw/while/Identity_4IdentityBrnn_classification/cell_1/bidirectional_rnn/fw/fw/while/Switch_4:1*
T0*
_output_shapes
:	
Â
=rnn_classification/cell_1/bidirectional_rnn/fw/fw/while/add/yConstA^rnn_classification/cell_1/bidirectional_rnn/fw/fw/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
ä
;rnn_classification/cell_1/bidirectional_rnn/fw/fw/while/addAdd@rnn_classification/cell_1/bidirectional_rnn/fw/fw/while/Identity=rnn_classification/cell_1/bidirectional_rnn/fw/fw/while/add/y*
T0*
_output_shapes
: 
ô
Irnn_classification/cell_1/bidirectional_rnn/fw/fw/while/TensorArrayReadV3TensorArrayReadV3Ornn_classification/cell_1/bidirectional_rnn/fw/fw/while/TensorArrayReadV3/EnterBrnn_classification/cell_1/bidirectional_rnn/fw/fw/while/Identity_1Qrnn_classification/cell_1/bidirectional_rnn/fw/fw/while/TensorArrayReadV3/Enter_1*
dtype0*
_output_shapes
:	
Ă
Ornn_classification/cell_1/bidirectional_rnn/fw/fw/while/TensorArrayReadV3/EnterEnter?rnn_classification/cell_1/bidirectional_rnn/fw/fw/TensorArray_1*
parallel_iterations *U

frame_nameGErnn_classification/cell_1/bidirectional_rnn/fw/fw/while/while_context*
_output_shapes
:*
T0*
is_constant(
î
Qrnn_classification/cell_1/bidirectional_rnn/fw/fw/while/TensorArrayReadV3/Enter_1Enterlrnn_classification/cell_1/bidirectional_rnn/fw/fw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3*
T0*
is_constant(*
parallel_iterations *U

frame_nameGErnn_classification/cell_1/bidirectional_rnn/fw/fw/while/while_context*
_output_shapes
: 

Drnn_classification/cell_1/bidirectional_rnn/fw/fw/while/GreaterEqualGreaterEqualBrnn_classification/cell_1/bidirectional_rnn/fw/fw/while/Identity_1Jrnn_classification/cell_1/bidirectional_rnn/fw/fw/while/GreaterEqual/Enter*
_output_shapes
:*
T0
ź
Jrnn_classification/cell_1/bidirectional_rnn/fw/fw/while/GreaterEqual/EnterEnter=rnn_classification/cell_1/bidirectional_rnn/fw/fw/CheckSeqLen*
T0*
is_constant(*
parallel_iterations *U

frame_nameGErnn_classification/cell_1/bidirectional_rnn/fw/fw/while/while_context*
_output_shapes
:

frnn_classification/cell_1/bidirectional_rnn/fw/basic_lstm_cell/kernel/Initializer/random_uniform/shapeConst*
valueB"     *X
_classN
LJloc:@rnn_classification/cell_1/bidirectional_rnn/fw/basic_lstm_cell/kernel*
dtype0*
_output_shapes
:

drnn_classification/cell_1/bidirectional_rnn/fw/basic_lstm_cell/kernel/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *
valueB
 *b§˝*X
_classN
LJloc:@rnn_classification/cell_1/bidirectional_rnn/fw/basic_lstm_cell/kernel

drnn_classification/cell_1/bidirectional_rnn/fw/basic_lstm_cell/kernel/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *
valueB
 *b§=*X
_classN
LJloc:@rnn_classification/cell_1/bidirectional_rnn/fw/basic_lstm_cell/kernel
ů
nrnn_classification/cell_1/bidirectional_rnn/fw/basic_lstm_cell/kernel/Initializer/random_uniform/RandomUniformRandomUniformfrnn_classification/cell_1/bidirectional_rnn/fw/basic_lstm_cell/kernel/Initializer/random_uniform/shape*
dtype0* 
_output_shapes
:
*
T0*X
_classN
LJloc:@rnn_classification/cell_1/bidirectional_rnn/fw/basic_lstm_cell/kernel
˛
drnn_classification/cell_1/bidirectional_rnn/fw/basic_lstm_cell/kernel/Initializer/random_uniform/subSubdrnn_classification/cell_1/bidirectional_rnn/fw/basic_lstm_cell/kernel/Initializer/random_uniform/maxdrnn_classification/cell_1/bidirectional_rnn/fw/basic_lstm_cell/kernel/Initializer/random_uniform/min*
T0*X
_classN
LJloc:@rnn_classification/cell_1/bidirectional_rnn/fw/basic_lstm_cell/kernel*
_output_shapes
: 
Ć
drnn_classification/cell_1/bidirectional_rnn/fw/basic_lstm_cell/kernel/Initializer/random_uniform/mulMulnrnn_classification/cell_1/bidirectional_rnn/fw/basic_lstm_cell/kernel/Initializer/random_uniform/RandomUniformdrnn_classification/cell_1/bidirectional_rnn/fw/basic_lstm_cell/kernel/Initializer/random_uniform/sub*
T0*X
_classN
LJloc:@rnn_classification/cell_1/bidirectional_rnn/fw/basic_lstm_cell/kernel* 
_output_shapes
:

¸
`rnn_classification/cell_1/bidirectional_rnn/fw/basic_lstm_cell/kernel/Initializer/random_uniformAdddrnn_classification/cell_1/bidirectional_rnn/fw/basic_lstm_cell/kernel/Initializer/random_uniform/muldrnn_classification/cell_1/bidirectional_rnn/fw/basic_lstm_cell/kernel/Initializer/random_uniform/min*
T0*X
_classN
LJloc:@rnn_classification/cell_1/bidirectional_rnn/fw/basic_lstm_cell/kernel* 
_output_shapes
:

ó
Ernn_classification/cell_1/bidirectional_rnn/fw/basic_lstm_cell/kernel
VariableV2*X
_classN
LJloc:@rnn_classification/cell_1/bidirectional_rnn/fw/basic_lstm_cell/kernel*
shape:
*
dtype0* 
_output_shapes
:


Lrnn_classification/cell_1/bidirectional_rnn/fw/basic_lstm_cell/kernel/AssignAssignErnn_classification/cell_1/bidirectional_rnn/fw/basic_lstm_cell/kernel`rnn_classification/cell_1/bidirectional_rnn/fw/basic_lstm_cell/kernel/Initializer/random_uniform*
T0*X
_classN
LJloc:@rnn_classification/cell_1/bidirectional_rnn/fw/basic_lstm_cell/kernel* 
_output_shapes
:

Č
Jrnn_classification/cell_1/bidirectional_rnn/fw/basic_lstm_cell/kernel/readIdentityErnn_classification/cell_1/bidirectional_rnn/fw/basic_lstm_cell/kernel*
T0* 
_output_shapes
:

ü
Urnn_classification/cell_1/bidirectional_rnn/fw/basic_lstm_cell/bias/Initializer/zerosConst*
valueB*    *V
_classL
JHloc:@rnn_classification/cell_1/bidirectional_rnn/fw/basic_lstm_cell/bias*
dtype0*
_output_shapes	
:
ĺ
Crnn_classification/cell_1/bidirectional_rnn/fw/basic_lstm_cell/bias
VariableV2*
dtype0*
_output_shapes	
:*V
_classL
JHloc:@rnn_classification/cell_1/bidirectional_rnn/fw/basic_lstm_cell/bias*
shape:
î
Jrnn_classification/cell_1/bidirectional_rnn/fw/basic_lstm_cell/bias/AssignAssignCrnn_classification/cell_1/bidirectional_rnn/fw/basic_lstm_cell/biasUrnn_classification/cell_1/bidirectional_rnn/fw/basic_lstm_cell/bias/Initializer/zeros*
T0*V
_classL
JHloc:@rnn_classification/cell_1/bidirectional_rnn/fw/basic_lstm_cell/bias*
_output_shapes	
:
ż
Hrnn_classification/cell_1/bidirectional_rnn/fw/basic_lstm_cell/bias/readIdentityCrnn_classification/cell_1/bidirectional_rnn/fw/basic_lstm_cell/bias*
T0*
_output_shapes	
:
Ň
Mrnn_classification/cell_1/bidirectional_rnn/fw/fw/while/basic_lstm_cell/ConstConstA^rnn_classification/cell_1/bidirectional_rnn/fw/fw/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
Ř
Srnn_classification/cell_1/bidirectional_rnn/fw/fw/while/basic_lstm_cell/concat/axisConstA^rnn_classification/cell_1/bidirectional_rnn/fw/fw/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
ń
Nrnn_classification/cell_1/bidirectional_rnn/fw/fw/while/basic_lstm_cell/concatConcatV2Irnn_classification/cell_1/bidirectional_rnn/fw/fw/while/TensorArrayReadV3Brnn_classification/cell_1/bidirectional_rnn/fw/fw/while/Identity_4Srnn_classification/cell_1/bidirectional_rnn/fw/fw/while/basic_lstm_cell/concat/axis*
T0*
N*
_output_shapes
:	
¨
Nrnn_classification/cell_1/bidirectional_rnn/fw/fw/while/basic_lstm_cell/MatMulMatMulNrnn_classification/cell_1/bidirectional_rnn/fw/fw/while/basic_lstm_cell/concatTrnn_classification/cell_1/bidirectional_rnn/fw/fw/while/basic_lstm_cell/MatMul/Enter*
_output_shapes
:	*
T0
Ů
Trnn_classification/cell_1/bidirectional_rnn/fw/fw/while/basic_lstm_cell/MatMul/EnterEnterJrnn_classification/cell_1/bidirectional_rnn/fw/basic_lstm_cell/kernel/read*
parallel_iterations *U

frame_nameGErnn_classification/cell_1/bidirectional_rnn/fw/fw/while/while_context* 
_output_shapes
:
*
T0*
is_constant(
Ť
Ornn_classification/cell_1/bidirectional_rnn/fw/fw/while/basic_lstm_cell/BiasAddBiasAddNrnn_classification/cell_1/bidirectional_rnn/fw/fw/while/basic_lstm_cell/MatMulUrnn_classification/cell_1/bidirectional_rnn/fw/fw/while/basic_lstm_cell/BiasAdd/Enter*
T0*
_output_shapes
:	
Ó
Urnn_classification/cell_1/bidirectional_rnn/fw/fw/while/basic_lstm_cell/BiasAdd/EnterEnterHrnn_classification/cell_1/bidirectional_rnn/fw/basic_lstm_cell/bias/read*
T0*
is_constant(*
parallel_iterations *U

frame_nameGErnn_classification/cell_1/bidirectional_rnn/fw/fw/while/while_context*
_output_shapes	
:
Ô
Ornn_classification/cell_1/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Const_1ConstA^rnn_classification/cell_1/bidirectional_rnn/fw/fw/while/Identity*
dtype0*
_output_shapes
: *
value	B :
Ň
Mrnn_classification/cell_1/bidirectional_rnn/fw/fw/while/basic_lstm_cell/splitSplitMrnn_classification/cell_1/bidirectional_rnn/fw/fw/while/basic_lstm_cell/ConstOrnn_classification/cell_1/bidirectional_rnn/fw/fw/while/basic_lstm_cell/BiasAdd*
T0*
	num_split*@
_output_shapes.
,:	:	:	:	
×
Ornn_classification/cell_1/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Const_2ConstA^rnn_classification/cell_1/bidirectional_rnn/fw/fw/while/Identity*
valueB
 *  ?*
dtype0*
_output_shapes
: 

Krnn_classification/cell_1/bidirectional_rnn/fw/fw/while/basic_lstm_cell/AddAddOrnn_classification/cell_1/bidirectional_rnn/fw/fw/while/basic_lstm_cell/split:2Ornn_classification/cell_1/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Const_2*
T0*
_output_shapes
:	
Ń
Ornn_classification/cell_1/bidirectional_rnn/fw/fw/while/basic_lstm_cell/SigmoidSigmoidKrnn_classification/cell_1/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Add*
T0*
_output_shapes
:	

Krnn_classification/cell_1/bidirectional_rnn/fw/fw/while/basic_lstm_cell/MulMulBrnn_classification/cell_1/bidirectional_rnn/fw/fw/while/Identity_3Ornn_classification/cell_1/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Sigmoid*
T0*
_output_shapes
:	
Ő
Qrnn_classification/cell_1/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Sigmoid_1SigmoidMrnn_classification/cell_1/bidirectional_rnn/fw/fw/while/basic_lstm_cell/split*
T0*
_output_shapes
:	
Ď
Lrnn_classification/cell_1/bidirectional_rnn/fw/fw/while/basic_lstm_cell/TanhTanhOrnn_classification/cell_1/bidirectional_rnn/fw/fw/while/basic_lstm_cell/split:1*
T0*
_output_shapes
:	

Mrnn_classification/cell_1/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_1MulQrnn_classification/cell_1/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Sigmoid_1Lrnn_classification/cell_1/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Tanh*
T0*
_output_shapes
:	

Mrnn_classification/cell_1/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Add_1AddKrnn_classification/cell_1/bidirectional_rnn/fw/fw/while/basic_lstm_cell/MulMrnn_classification/cell_1/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_1*
T0*
_output_shapes
:	
Ď
Nrnn_classification/cell_1/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Tanh_1TanhMrnn_classification/cell_1/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Add_1*
T0*
_output_shapes
:	
×
Qrnn_classification/cell_1/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Sigmoid_2SigmoidOrnn_classification/cell_1/bidirectional_rnn/fw/fw/while/basic_lstm_cell/split:3*
T0*
_output_shapes
:	
Ą
Mrnn_classification/cell_1/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_2MulNrnn_classification/cell_1/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Tanh_1Qrnn_classification/cell_1/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Sigmoid_2*
_output_shapes
:	*
T0
Ż
>rnn_classification/cell_1/bidirectional_rnn/fw/fw/while/SelectSelectDrnn_classification/cell_1/bidirectional_rnn/fw/fw/while/GreaterEqualDrnn_classification/cell_1/bidirectional_rnn/fw/fw/while/Select/EnterMrnn_classification/cell_1/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_2*
T0*`
_classV
TRloc:@rnn_classification/cell_1/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_2*
_output_shapes
:	

Drnn_classification/cell_1/bidirectional_rnn/fw/fw/while/Select/EnterEnter7rnn_classification/cell_1/bidirectional_rnn/fw/fw/zeros*
is_constant(*
_output_shapes
:	*U

frame_nameGErnn_classification/cell_1/bidirectional_rnn/fw/fw/while/while_context*
T0*`
_classV
TRloc:@rnn_classification/cell_1/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_2*
parallel_iterations 
Ż
@rnn_classification/cell_1/bidirectional_rnn/fw/fw/while/Select_1SelectDrnn_classification/cell_1/bidirectional_rnn/fw/fw/while/GreaterEqualBrnn_classification/cell_1/bidirectional_rnn/fw/fw/while/Identity_3Mrnn_classification/cell_1/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Add_1*
_output_shapes
:	*
T0*`
_classV
TRloc:@rnn_classification/cell_1/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Add_1
Ż
@rnn_classification/cell_1/bidirectional_rnn/fw/fw/while/Select_2SelectDrnn_classification/cell_1/bidirectional_rnn/fw/fw/while/GreaterEqualBrnn_classification/cell_1/bidirectional_rnn/fw/fw/while/Identity_4Mrnn_classification/cell_1/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_2*
T0*`
_classV
TRloc:@rnn_classification/cell_1/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_2*
_output_shapes
:	

[rnn_classification/cell_1/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3TensorArrayWriteV3arnn_classification/cell_1/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3/EnterBrnn_classification/cell_1/bidirectional_rnn/fw/fw/while/Identity_1>rnn_classification/cell_1/bidirectional_rnn/fw/fw/while/SelectBrnn_classification/cell_1/bidirectional_rnn/fw/fw/while/Identity_2*
T0*`
_classV
TRloc:@rnn_classification/cell_1/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_2*
_output_shapes
: 
ľ
arnn_classification/cell_1/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3/EnterEnter=rnn_classification/cell_1/bidirectional_rnn/fw/fw/TensorArray*
T0*`
_classV
TRloc:@rnn_classification/cell_1/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_2*
parallel_iterations *
is_constant(*
_output_shapes
:*U

frame_nameGErnn_classification/cell_1/bidirectional_rnn/fw/fw/while/while_context
Ä
?rnn_classification/cell_1/bidirectional_rnn/fw/fw/while/add_1/yConstA^rnn_classification/cell_1/bidirectional_rnn/fw/fw/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
ę
=rnn_classification/cell_1/bidirectional_rnn/fw/fw/while/add_1AddBrnn_classification/cell_1/bidirectional_rnn/fw/fw/while/Identity_1?rnn_classification/cell_1/bidirectional_rnn/fw/fw/while/add_1/y*
T0*
_output_shapes
: 
´
Ernn_classification/cell_1/bidirectional_rnn/fw/fw/while/NextIterationNextIteration;rnn_classification/cell_1/bidirectional_rnn/fw/fw/while/add*
T0*
_output_shapes
: 
¸
Grnn_classification/cell_1/bidirectional_rnn/fw/fw/while/NextIteration_1NextIteration=rnn_classification/cell_1/bidirectional_rnn/fw/fw/while/add_1*
T0*
_output_shapes
: 
Ö
Grnn_classification/cell_1/bidirectional_rnn/fw/fw/while/NextIteration_2NextIteration[rnn_classification/cell_1/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3*
T0*
_output_shapes
: 
Ä
Grnn_classification/cell_1/bidirectional_rnn/fw/fw/while/NextIteration_3NextIteration@rnn_classification/cell_1/bidirectional_rnn/fw/fw/while/Select_1*
T0*
_output_shapes
:	
Ä
Grnn_classification/cell_1/bidirectional_rnn/fw/fw/while/NextIteration_4NextIteration@rnn_classification/cell_1/bidirectional_rnn/fw/fw/while/Select_2*
T0*
_output_shapes
:	
Ľ
<rnn_classification/cell_1/bidirectional_rnn/fw/fw/while/ExitExit>rnn_classification/cell_1/bidirectional_rnn/fw/fw/while/Switch*
T0*
_output_shapes
: 
Š
>rnn_classification/cell_1/bidirectional_rnn/fw/fw/while/Exit_1Exit@rnn_classification/cell_1/bidirectional_rnn/fw/fw/while/Switch_1*
T0*
_output_shapes
: 
Š
>rnn_classification/cell_1/bidirectional_rnn/fw/fw/while/Exit_2Exit@rnn_classification/cell_1/bidirectional_rnn/fw/fw/while/Switch_2*
T0*
_output_shapes
: 
˛
>rnn_classification/cell_1/bidirectional_rnn/fw/fw/while/Exit_3Exit@rnn_classification/cell_1/bidirectional_rnn/fw/fw/while/Switch_3*
T0*
_output_shapes
:	
˛
>rnn_classification/cell_1/bidirectional_rnn/fw/fw/while/Exit_4Exit@rnn_classification/cell_1/bidirectional_rnn/fw/fw/while/Switch_4*
T0*
_output_shapes
:	
Ň
Trnn_classification/cell_1/bidirectional_rnn/fw/fw/TensorArrayStack/TensorArraySizeV3TensorArraySizeV3=rnn_classification/cell_1/bidirectional_rnn/fw/fw/TensorArray>rnn_classification/cell_1/bidirectional_rnn/fw/fw/while/Exit_2*P
_classF
DBloc:@rnn_classification/cell_1/bidirectional_rnn/fw/fw/TensorArray*
_output_shapes
: 
â
Nrnn_classification/cell_1/bidirectional_rnn/fw/fw/TensorArrayStack/range/startConst*
value	B : *P
_classF
DBloc:@rnn_classification/cell_1/bidirectional_rnn/fw/fw/TensorArray*
dtype0*
_output_shapes
: 
â
Nrnn_classification/cell_1/bidirectional_rnn/fw/fw/TensorArrayStack/range/deltaConst*
dtype0*
_output_shapes
: *
value	B :*P
_classF
DBloc:@rnn_classification/cell_1/bidirectional_rnn/fw/fw/TensorArray
ž
Hrnn_classification/cell_1/bidirectional_rnn/fw/fw/TensorArrayStack/rangeRangeNrnn_classification/cell_1/bidirectional_rnn/fw/fw/TensorArrayStack/range/startTrnn_classification/cell_1/bidirectional_rnn/fw/fw/TensorArrayStack/TensorArraySizeV3Nrnn_classification/cell_1/bidirectional_rnn/fw/fw/TensorArrayStack/range/delta*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*P
_classF
DBloc:@rnn_classification/cell_1/bidirectional_rnn/fw/fw/TensorArray
á
Vrnn_classification/cell_1/bidirectional_rnn/fw/fw/TensorArrayStack/TensorArrayGatherV3TensorArrayGatherV3=rnn_classification/cell_1/bidirectional_rnn/fw/fw/TensorArrayHrnn_classification/cell_1/bidirectional_rnn/fw/fw/TensorArrayStack/range>rnn_classification/cell_1/bidirectional_rnn/fw/fw/while/Exit_2*P
_classF
DBloc:@rnn_classification/cell_1/bidirectional_rnn/fw/fw/TensorArray*
dtype0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙*
element_shape:	

9rnn_classification/cell_1/bidirectional_rnn/fw/fw/Const_5Const*
valueB:*
dtype0*
_output_shapes
:
z
8rnn_classification/cell_1/bidirectional_rnn/fw/fw/Rank_1Const*
value	B :*
dtype0*
_output_shapes
: 

?rnn_classification/cell_1/bidirectional_rnn/fw/fw/range_1/startConst*
value	B :*
dtype0*
_output_shapes
: 

?rnn_classification/cell_1/bidirectional_rnn/fw/fw/range_1/deltaConst*
value	B :*
dtype0*
_output_shapes
: 

9rnn_classification/cell_1/bidirectional_rnn/fw/fw/range_1Range?rnn_classification/cell_1/bidirectional_rnn/fw/fw/range_1/start8rnn_classification/cell_1/bidirectional_rnn/fw/fw/Rank_1?rnn_classification/cell_1/bidirectional_rnn/fw/fw/range_1/delta*
_output_shapes
:

Crnn_classification/cell_1/bidirectional_rnn/fw/fw/concat_2/values_0Const*
valueB"       *
dtype0*
_output_shapes
:

?rnn_classification/cell_1/bidirectional_rnn/fw/fw/concat_2/axisConst*
dtype0*
_output_shapes
: *
value	B : 
ľ
:rnn_classification/cell_1/bidirectional_rnn/fw/fw/concat_2ConcatV2Crnn_classification/cell_1/bidirectional_rnn/fw/fw/concat_2/values_09rnn_classification/cell_1/bidirectional_rnn/fw/fw/range_1?rnn_classification/cell_1/bidirectional_rnn/fw/fw/concat_2/axis*
T0*
N*
_output_shapes
:

=rnn_classification/cell_1/bidirectional_rnn/fw/fw/transpose_1	TransposeVrnn_classification/cell_1/bidirectional_rnn/fw/fw/TensorArrayStack/TensorArrayGatherV3:rnn_classification/cell_1/bidirectional_rnn/fw/fw/concat_2*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙
Â
>rnn_classification/cell_1/bidirectional_rnn/bw/ReverseSequenceReverseSequence rnn_classification/cell_0/concatSqueeze*
T0*
seq_dim*,
_output_shapes
:˙˙˙˙˙˙˙˙˙
x
6rnn_classification/cell_1/bidirectional_rnn/bw/bw/RankConst*
value	B :*
dtype0*
_output_shapes
: 

=rnn_classification/cell_1/bidirectional_rnn/bw/bw/range/startConst*
value	B :*
dtype0*
_output_shapes
: 

=rnn_classification/cell_1/bidirectional_rnn/bw/bw/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 

7rnn_classification/cell_1/bidirectional_rnn/bw/bw/rangeRange=rnn_classification/cell_1/bidirectional_rnn/bw/bw/range/start6rnn_classification/cell_1/bidirectional_rnn/bw/bw/Rank=rnn_classification/cell_1/bidirectional_rnn/bw/bw/range/delta*
_output_shapes
:

Arnn_classification/cell_1/bidirectional_rnn/bw/bw/concat/values_0Const*
valueB"       *
dtype0*
_output_shapes
:

=rnn_classification/cell_1/bidirectional_rnn/bw/bw/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
­
8rnn_classification/cell_1/bidirectional_rnn/bw/bw/concatConcatV2Arnn_classification/cell_1/bidirectional_rnn/bw/bw/concat/values_07rnn_classification/cell_1/bidirectional_rnn/bw/bw/range=rnn_classification/cell_1/bidirectional_rnn/bw/bw/concat/axis*
N*
_output_shapes
:*
T0
ů
;rnn_classification/cell_1/bidirectional_rnn/bw/bw/transpose	Transpose>rnn_classification/cell_1/bidirectional_rnn/bw/ReverseSequence8rnn_classification/cell_1/bidirectional_rnn/bw/bw/concat*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙
~
9rnn_classification/cell_1/bidirectional_rnn/bw/bw/ToInt32CastSqueeze*

SrcT0	*

DstT0*
_output_shapes
:
­
Arnn_classification/cell_1/bidirectional_rnn/bw/bw/sequence_lengthIdentity9rnn_classification/cell_1/bidirectional_rnn/bw/bw/ToInt32*
_output_shapes
:*
T0
°
frnn_classification/cell_1/bidirectional_rnn/bw/bw/DropoutWrapperZeroState/BasicLSTMCellZeroState/ConstConst*
dtype0*
_output_shapes
:*
valueB:
ł
hrnn_classification/cell_1/bidirectional_rnn/bw/bw/DropoutWrapperZeroState/BasicLSTMCellZeroState/Const_1Const*
valueB:*
dtype0*
_output_shapes
:
Ž
lrnn_classification/cell_1/bidirectional_rnn/bw/bw/DropoutWrapperZeroState/BasicLSTMCellZeroState/concat/axisConst*
dtype0*
_output_shapes
: *
value	B : 
á
grnn_classification/cell_1/bidirectional_rnn/bw/bw/DropoutWrapperZeroState/BasicLSTMCellZeroState/concatConcatV2frnn_classification/cell_1/bidirectional_rnn/bw/bw/DropoutWrapperZeroState/BasicLSTMCellZeroState/Consthrnn_classification/cell_1/bidirectional_rnn/bw/bw/DropoutWrapperZeroState/BasicLSTMCellZeroState/Const_1lrnn_classification/cell_1/bidirectional_rnn/bw/bw/DropoutWrapperZeroState/BasicLSTMCellZeroState/concat/axis*
T0*
N*
_output_shapes
:
ą
lrnn_classification/cell_1/bidirectional_rnn/bw/bw/DropoutWrapperZeroState/BasicLSTMCellZeroState/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
ď
frnn_classification/cell_1/bidirectional_rnn/bw/bw/DropoutWrapperZeroState/BasicLSTMCellZeroState/zerosFillgrnn_classification/cell_1/bidirectional_rnn/bw/bw/DropoutWrapperZeroState/BasicLSTMCellZeroState/concatlrnn_classification/cell_1/bidirectional_rnn/bw/bw/DropoutWrapperZeroState/BasicLSTMCellZeroState/zeros/Const*
T0*
_output_shapes
:	
˛
hrnn_classification/cell_1/bidirectional_rnn/bw/bw/DropoutWrapperZeroState/BasicLSTMCellZeroState/Const_2Const*
valueB:*
dtype0*
_output_shapes
:
ł
hrnn_classification/cell_1/bidirectional_rnn/bw/bw/DropoutWrapperZeroState/BasicLSTMCellZeroState/Const_3Const*
valueB:*
dtype0*
_output_shapes
:
˛
hrnn_classification/cell_1/bidirectional_rnn/bw/bw/DropoutWrapperZeroState/BasicLSTMCellZeroState/Const_4Const*
dtype0*
_output_shapes
:*
valueB:
ł
hrnn_classification/cell_1/bidirectional_rnn/bw/bw/DropoutWrapperZeroState/BasicLSTMCellZeroState/Const_5Const*
valueB:*
dtype0*
_output_shapes
:
°
nrnn_classification/cell_1/bidirectional_rnn/bw/bw/DropoutWrapperZeroState/BasicLSTMCellZeroState/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
ç
irnn_classification/cell_1/bidirectional_rnn/bw/bw/DropoutWrapperZeroState/BasicLSTMCellZeroState/concat_1ConcatV2hrnn_classification/cell_1/bidirectional_rnn/bw/bw/DropoutWrapperZeroState/BasicLSTMCellZeroState/Const_4hrnn_classification/cell_1/bidirectional_rnn/bw/bw/DropoutWrapperZeroState/BasicLSTMCellZeroState/Const_5nrnn_classification/cell_1/bidirectional_rnn/bw/bw/DropoutWrapperZeroState/BasicLSTMCellZeroState/concat_1/axis*
T0*
N*
_output_shapes
:
ł
nrnn_classification/cell_1/bidirectional_rnn/bw/bw/DropoutWrapperZeroState/BasicLSTMCellZeroState/zeros_1/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
ő
hrnn_classification/cell_1/bidirectional_rnn/bw/bw/DropoutWrapperZeroState/BasicLSTMCellZeroState/zeros_1Fillirnn_classification/cell_1/bidirectional_rnn/bw/bw/DropoutWrapperZeroState/BasicLSTMCellZeroState/concat_1nrnn_classification/cell_1/bidirectional_rnn/bw/bw/DropoutWrapperZeroState/BasicLSTMCellZeroState/zeros_1/Const*
T0*
_output_shapes
:	
˛
hrnn_classification/cell_1/bidirectional_rnn/bw/bw/DropoutWrapperZeroState/BasicLSTMCellZeroState/Const_6Const*
valueB:*
dtype0*
_output_shapes
:
ł
hrnn_classification/cell_1/bidirectional_rnn/bw/bw/DropoutWrapperZeroState/BasicLSTMCellZeroState/Const_7Const*
valueB:*
dtype0*
_output_shapes
:

7rnn_classification/cell_1/bidirectional_rnn/bw/bw/ShapeConst*
valueB:*
dtype0*
_output_shapes
:

7rnn_classification/cell_1/bidirectional_rnn/bw/bw/stackConst*
valueB:*
dtype0*
_output_shapes
:
×
7rnn_classification/cell_1/bidirectional_rnn/bw/bw/EqualEqual7rnn_classification/cell_1/bidirectional_rnn/bw/bw/Shape7rnn_classification/cell_1/bidirectional_rnn/bw/bw/stack*
T0*
_output_shapes
:

7rnn_classification/cell_1/bidirectional_rnn/bw/bw/ConstConst*
valueB: *
dtype0*
_output_shapes
:
Ć
5rnn_classification/cell_1/bidirectional_rnn/bw/bw/AllAll7rnn_classification/cell_1/bidirectional_rnn/bw/bw/Equal7rnn_classification/cell_1/bidirectional_rnn/bw/bw/Const*
_output_shapes
: 
ŕ
>rnn_classification/cell_1/bidirectional_rnn/bw/bw/Assert/ConstConst*r
valueiBg BaExpected shape for Tensor rnn_classification/cell_1/bidirectional_rnn/bw/bw/sequence_length:0 is *
dtype0*
_output_shapes
: 

@rnn_classification/cell_1/bidirectional_rnn/bw/bw/Assert/Const_1Const*!
valueB B but saw shape: *
dtype0*
_output_shapes
: 
č
Frnn_classification/cell_1/bidirectional_rnn/bw/bw/Assert/Assert/data_0Const*r
valueiBg BaExpected shape for Tensor rnn_classification/cell_1/bidirectional_rnn/bw/bw/sequence_length:0 is *
dtype0*
_output_shapes
: 

Frnn_classification/cell_1/bidirectional_rnn/bw/bw/Assert/Assert/data_2Const*!
valueB B but saw shape: *
dtype0*
_output_shapes
: 

?rnn_classification/cell_1/bidirectional_rnn/bw/bw/Assert/AssertAssert5rnn_classification/cell_1/bidirectional_rnn/bw/bw/AllFrnn_classification/cell_1/bidirectional_rnn/bw/bw/Assert/Assert/data_07rnn_classification/cell_1/bidirectional_rnn/bw/bw/stackFrnn_classification/cell_1/bidirectional_rnn/bw/bw/Assert/Assert/data_27rnn_classification/cell_1/bidirectional_rnn/bw/bw/Shape*
T
2
ó
=rnn_classification/cell_1/bidirectional_rnn/bw/bw/CheckSeqLenIdentityArnn_classification/cell_1/bidirectional_rnn/bw/bw/sequence_length@^rnn_classification/cell_1/bidirectional_rnn/bw/bw/Assert/Assert*
T0*
_output_shapes
:
¤
9rnn_classification/cell_1/bidirectional_rnn/bw/bw/Shape_1Shape;rnn_classification/cell_1/bidirectional_rnn/bw/bw/transpose*
T0*
_output_shapes
:

Ernn_classification/cell_1/bidirectional_rnn/bw/bw/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:

Grnn_classification/cell_1/bidirectional_rnn/bw/bw/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:

Grnn_classification/cell_1/bidirectional_rnn/bw/bw/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
Š
?rnn_classification/cell_1/bidirectional_rnn/bw/bw/strided_sliceStridedSlice9rnn_classification/cell_1/bidirectional_rnn/bw/bw/Shape_1Ernn_classification/cell_1/bidirectional_rnn/bw/bw/strided_slice/stackGrnn_classification/cell_1/bidirectional_rnn/bw/bw/strided_slice/stack_1Grnn_classification/cell_1/bidirectional_rnn/bw/bw/strided_slice/stack_2*
Index0*
T0*
shrink_axis_mask*
_output_shapes
: 

9rnn_classification/cell_1/bidirectional_rnn/bw/bw/Const_1Const*
valueB:*
dtype0*
_output_shapes
:

9rnn_classification/cell_1/bidirectional_rnn/bw/bw/Const_2Const*
valueB:*
dtype0*
_output_shapes
:

?rnn_classification/cell_1/bidirectional_rnn/bw/bw/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
Ť
:rnn_classification/cell_1/bidirectional_rnn/bw/bw/concat_1ConcatV29rnn_classification/cell_1/bidirectional_rnn/bw/bw/Const_19rnn_classification/cell_1/bidirectional_rnn/bw/bw/Const_2?rnn_classification/cell_1/bidirectional_rnn/bw/bw/concat_1/axis*
T0*
N*
_output_shapes
:

=rnn_classification/cell_1/bidirectional_rnn/bw/bw/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
ä
7rnn_classification/cell_1/bidirectional_rnn/bw/bw/zerosFill:rnn_classification/cell_1/bidirectional_rnn/bw/bw/concat_1=rnn_classification/cell_1/bidirectional_rnn/bw/bw/zeros/Const*
T0*
_output_shapes
:	

9rnn_classification/cell_1/bidirectional_rnn/bw/bw/Const_3Const*
valueB: *
dtype0*
_output_shapes
:
×
5rnn_classification/cell_1/bidirectional_rnn/bw/bw/MinMin=rnn_classification/cell_1/bidirectional_rnn/bw/bw/CheckSeqLen9rnn_classification/cell_1/bidirectional_rnn/bw/bw/Const_3*
T0*
_output_shapes
: 

9rnn_classification/cell_1/bidirectional_rnn/bw/bw/Const_4Const*
dtype0*
_output_shapes
:*
valueB: 
×
5rnn_classification/cell_1/bidirectional_rnn/bw/bw/MaxMax=rnn_classification/cell_1/bidirectional_rnn/bw/bw/CheckSeqLen9rnn_classification/cell_1/bidirectional_rnn/bw/bw/Const_4*
_output_shapes
: *
T0
x
6rnn_classification/cell_1/bidirectional_rnn/bw/bw/timeConst*
value	B : *
dtype0*
_output_shapes
: 
×
=rnn_classification/cell_1/bidirectional_rnn/bw/bw/TensorArrayTensorArrayV3?rnn_classification/cell_1/bidirectional_rnn/bw/bw/strided_slice*
identical_element_shapes(*]
tensor_array_nameHFrnn_classification/cell_1/bidirectional_rnn/bw/bw/dynamic_rnn/output_0*
dtype0*
_output_shapes

:: *
element_shape:	
Ř
?rnn_classification/cell_1/bidirectional_rnn/bw/bw/TensorArray_1TensorArrayV3?rnn_classification/cell_1/bidirectional_rnn/bw/bw/strided_slice*
element_shape:	*
identical_element_shapes(*\
tensor_array_nameGErnn_classification/cell_1/bidirectional_rnn/bw/bw/dynamic_rnn/input_0*
dtype0*
_output_shapes

:: 
ľ
Jrnn_classification/cell_1/bidirectional_rnn/bw/bw/TensorArrayUnstack/ShapeShape;rnn_classification/cell_1/bidirectional_rnn/bw/bw/transpose*
T0*
_output_shapes
:
˘
Xrnn_classification/cell_1/bidirectional_rnn/bw/bw/TensorArrayUnstack/strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB: 
¤
Zrnn_classification/cell_1/bidirectional_rnn/bw/bw/TensorArrayUnstack/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
¤
Zrnn_classification/cell_1/bidirectional_rnn/bw/bw/TensorArrayUnstack/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:

Rrnn_classification/cell_1/bidirectional_rnn/bw/bw/TensorArrayUnstack/strided_sliceStridedSliceJrnn_classification/cell_1/bidirectional_rnn/bw/bw/TensorArrayUnstack/ShapeXrnn_classification/cell_1/bidirectional_rnn/bw/bw/TensorArrayUnstack/strided_slice/stackZrnn_classification/cell_1/bidirectional_rnn/bw/bw/TensorArrayUnstack/strided_slice/stack_1Zrnn_classification/cell_1/bidirectional_rnn/bw/bw/TensorArrayUnstack/strided_slice/stack_2*
T0*
Index0*
shrink_axis_mask*
_output_shapes
: 

Prnn_classification/cell_1/bidirectional_rnn/bw/bw/TensorArrayUnstack/range/startConst*
value	B : *
dtype0*
_output_shapes
: 

Prnn_classification/cell_1/bidirectional_rnn/bw/bw/TensorArrayUnstack/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
đ
Jrnn_classification/cell_1/bidirectional_rnn/bw/bw/TensorArrayUnstack/rangeRangePrnn_classification/cell_1/bidirectional_rnn/bw/bw/TensorArrayUnstack/range/startRrnn_classification/cell_1/bidirectional_rnn/bw/bw/TensorArrayUnstack/strided_slicePrnn_classification/cell_1/bidirectional_rnn/bw/bw/TensorArrayUnstack/range/delta*#
_output_shapes
:˙˙˙˙˙˙˙˙˙

lrnn_classification/cell_1/bidirectional_rnn/bw/bw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3TensorArrayScatterV3?rnn_classification/cell_1/bidirectional_rnn/bw/bw/TensorArray_1Jrnn_classification/cell_1/bidirectional_rnn/bw/bw/TensorArrayUnstack/range;rnn_classification/cell_1/bidirectional_rnn/bw/bw/transposeArnn_classification/cell_1/bidirectional_rnn/bw/bw/TensorArray_1:1*
T0*N
_classD
B@loc:@rnn_classification/cell_1/bidirectional_rnn/bw/bw/transpose*
_output_shapes
: 
}
;rnn_classification/cell_1/bidirectional_rnn/bw/bw/Maximum/xConst*
value	B :*
dtype0*
_output_shapes
: 
Ů
9rnn_classification/cell_1/bidirectional_rnn/bw/bw/MaximumMaximum;rnn_classification/cell_1/bidirectional_rnn/bw/bw/Maximum/x5rnn_classification/cell_1/bidirectional_rnn/bw/bw/Max*
_output_shapes
: *
T0
á
9rnn_classification/cell_1/bidirectional_rnn/bw/bw/MinimumMinimum?rnn_classification/cell_1/bidirectional_rnn/bw/bw/strided_slice9rnn_classification/cell_1/bidirectional_rnn/bw/bw/Maximum*
T0*
_output_shapes
: 

Irnn_classification/cell_1/bidirectional_rnn/bw/bw/while/iteration_counterConst*
dtype0*
_output_shapes
: *
value	B : 
¤
=rnn_classification/cell_1/bidirectional_rnn/bw/bw/while/EnterEnterIrnn_classification/cell_1/bidirectional_rnn/bw/bw/while/iteration_counter*
parallel_iterations *U

frame_nameGErnn_classification/cell_1/bidirectional_rnn/bw/bw/while/while_context*
_output_shapes
: *
T0

?rnn_classification/cell_1/bidirectional_rnn/bw/bw/while/Enter_1Enter6rnn_classification/cell_1/bidirectional_rnn/bw/bw/time*
T0*
parallel_iterations *U

frame_nameGErnn_classification/cell_1/bidirectional_rnn/bw/bw/while/while_context*
_output_shapes
: 

?rnn_classification/cell_1/bidirectional_rnn/bw/bw/while/Enter_2Enter?rnn_classification/cell_1/bidirectional_rnn/bw/bw/TensorArray:1*
T0*
parallel_iterations *U

frame_nameGErnn_classification/cell_1/bidirectional_rnn/bw/bw/while/while_context*
_output_shapes
: 
Ě
?rnn_classification/cell_1/bidirectional_rnn/bw/bw/while/Enter_3Enterfrnn_classification/cell_1/bidirectional_rnn/bw/bw/DropoutWrapperZeroState/BasicLSTMCellZeroState/zeros*
parallel_iterations *U

frame_nameGErnn_classification/cell_1/bidirectional_rnn/bw/bw/while/while_context*
_output_shapes
:	*
T0
Î
?rnn_classification/cell_1/bidirectional_rnn/bw/bw/while/Enter_4Enterhrnn_classification/cell_1/bidirectional_rnn/bw/bw/DropoutWrapperZeroState/BasicLSTMCellZeroState/zeros_1*
T0*
parallel_iterations *U

frame_nameGErnn_classification/cell_1/bidirectional_rnn/bw/bw/while/while_context*
_output_shapes
:	
ř
=rnn_classification/cell_1/bidirectional_rnn/bw/bw/while/MergeMerge=rnn_classification/cell_1/bidirectional_rnn/bw/bw/while/EnterErnn_classification/cell_1/bidirectional_rnn/bw/bw/while/NextIteration*
T0*
N*
_output_shapes
: : 
ţ
?rnn_classification/cell_1/bidirectional_rnn/bw/bw/while/Merge_1Merge?rnn_classification/cell_1/bidirectional_rnn/bw/bw/while/Enter_1Grnn_classification/cell_1/bidirectional_rnn/bw/bw/while/NextIteration_1*
T0*
N*
_output_shapes
: : 
ţ
?rnn_classification/cell_1/bidirectional_rnn/bw/bw/while/Merge_2Merge?rnn_classification/cell_1/bidirectional_rnn/bw/bw/while/Enter_2Grnn_classification/cell_1/bidirectional_rnn/bw/bw/while/NextIteration_2*
T0*
N*
_output_shapes
: : 

?rnn_classification/cell_1/bidirectional_rnn/bw/bw/while/Merge_3Merge?rnn_classification/cell_1/bidirectional_rnn/bw/bw/while/Enter_3Grnn_classification/cell_1/bidirectional_rnn/bw/bw/while/NextIteration_3*
N*!
_output_shapes
:	: *
T0

?rnn_classification/cell_1/bidirectional_rnn/bw/bw/while/Merge_4Merge?rnn_classification/cell_1/bidirectional_rnn/bw/bw/while/Enter_4Grnn_classification/cell_1/bidirectional_rnn/bw/bw/while/NextIteration_4*
T0*
N*!
_output_shapes
:	: 
č
<rnn_classification/cell_1/bidirectional_rnn/bw/bw/while/LessLess=rnn_classification/cell_1/bidirectional_rnn/bw/bw/while/MergeBrnn_classification/cell_1/bidirectional_rnn/bw/bw/while/Less/Enter*
T0*
_output_shapes
: 
˛
Brnn_classification/cell_1/bidirectional_rnn/bw/bw/while/Less/EnterEnter?rnn_classification/cell_1/bidirectional_rnn/bw/bw/strided_slice*
T0*
is_constant(*
parallel_iterations *U

frame_nameGErnn_classification/cell_1/bidirectional_rnn/bw/bw/while/while_context*
_output_shapes
: 
î
>rnn_classification/cell_1/bidirectional_rnn/bw/bw/while/Less_1Less?rnn_classification/cell_1/bidirectional_rnn/bw/bw/while/Merge_1Drnn_classification/cell_1/bidirectional_rnn/bw/bw/while/Less_1/Enter*
_output_shapes
: *
T0
Ž
Drnn_classification/cell_1/bidirectional_rnn/bw/bw/while/Less_1/EnterEnter9rnn_classification/cell_1/bidirectional_rnn/bw/bw/Minimum*
T0*
is_constant(*
parallel_iterations *U

frame_nameGErnn_classification/cell_1/bidirectional_rnn/bw/bw/while/while_context*
_output_shapes
: 
ć
Brnn_classification/cell_1/bidirectional_rnn/bw/bw/while/LogicalAnd
LogicalAnd<rnn_classification/cell_1/bidirectional_rnn/bw/bw/while/Less>rnn_classification/cell_1/bidirectional_rnn/bw/bw/while/Less_1*
_output_shapes
: 
¨
@rnn_classification/cell_1/bidirectional_rnn/bw/bw/while/LoopCondLoopCondBrnn_classification/cell_1/bidirectional_rnn/bw/bw/while/LogicalAnd*
_output_shapes
: 
ž
>rnn_classification/cell_1/bidirectional_rnn/bw/bw/while/SwitchSwitch=rnn_classification/cell_1/bidirectional_rnn/bw/bw/while/Merge@rnn_classification/cell_1/bidirectional_rnn/bw/bw/while/LoopCond*
T0*P
_classF
DBloc:@rnn_classification/cell_1/bidirectional_rnn/bw/bw/while/Merge*
_output_shapes
: : 
Ä
@rnn_classification/cell_1/bidirectional_rnn/bw/bw/while/Switch_1Switch?rnn_classification/cell_1/bidirectional_rnn/bw/bw/while/Merge_1@rnn_classification/cell_1/bidirectional_rnn/bw/bw/while/LoopCond*
_output_shapes
: : *
T0*R
_classH
FDloc:@rnn_classification/cell_1/bidirectional_rnn/bw/bw/while/Merge_1
Ä
@rnn_classification/cell_1/bidirectional_rnn/bw/bw/while/Switch_2Switch?rnn_classification/cell_1/bidirectional_rnn/bw/bw/while/Merge_2@rnn_classification/cell_1/bidirectional_rnn/bw/bw/while/LoopCond*
T0*R
_classH
FDloc:@rnn_classification/cell_1/bidirectional_rnn/bw/bw/while/Merge_2*
_output_shapes
: : 
Ö
@rnn_classification/cell_1/bidirectional_rnn/bw/bw/while/Switch_3Switch?rnn_classification/cell_1/bidirectional_rnn/bw/bw/while/Merge_3@rnn_classification/cell_1/bidirectional_rnn/bw/bw/while/LoopCond*
T0*R
_classH
FDloc:@rnn_classification/cell_1/bidirectional_rnn/bw/bw/while/Merge_3**
_output_shapes
:	:	
Ö
@rnn_classification/cell_1/bidirectional_rnn/bw/bw/while/Switch_4Switch?rnn_classification/cell_1/bidirectional_rnn/bw/bw/while/Merge_4@rnn_classification/cell_1/bidirectional_rnn/bw/bw/while/LoopCond*
T0*R
_classH
FDloc:@rnn_classification/cell_1/bidirectional_rnn/bw/bw/while/Merge_4**
_output_shapes
:	:	
Ż
@rnn_classification/cell_1/bidirectional_rnn/bw/bw/while/IdentityIdentity@rnn_classification/cell_1/bidirectional_rnn/bw/bw/while/Switch:1*
T0*
_output_shapes
: 
ł
Brnn_classification/cell_1/bidirectional_rnn/bw/bw/while/Identity_1IdentityBrnn_classification/cell_1/bidirectional_rnn/bw/bw/while/Switch_1:1*
T0*
_output_shapes
: 
ł
Brnn_classification/cell_1/bidirectional_rnn/bw/bw/while/Identity_2IdentityBrnn_classification/cell_1/bidirectional_rnn/bw/bw/while/Switch_2:1*
T0*
_output_shapes
: 
ź
Brnn_classification/cell_1/bidirectional_rnn/bw/bw/while/Identity_3IdentityBrnn_classification/cell_1/bidirectional_rnn/bw/bw/while/Switch_3:1*
T0*
_output_shapes
:	
ź
Brnn_classification/cell_1/bidirectional_rnn/bw/bw/while/Identity_4IdentityBrnn_classification/cell_1/bidirectional_rnn/bw/bw/while/Switch_4:1*
T0*
_output_shapes
:	
Â
=rnn_classification/cell_1/bidirectional_rnn/bw/bw/while/add/yConstA^rnn_classification/cell_1/bidirectional_rnn/bw/bw/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
ä
;rnn_classification/cell_1/bidirectional_rnn/bw/bw/while/addAdd@rnn_classification/cell_1/bidirectional_rnn/bw/bw/while/Identity=rnn_classification/cell_1/bidirectional_rnn/bw/bw/while/add/y*
T0*
_output_shapes
: 
ô
Irnn_classification/cell_1/bidirectional_rnn/bw/bw/while/TensorArrayReadV3TensorArrayReadV3Ornn_classification/cell_1/bidirectional_rnn/bw/bw/while/TensorArrayReadV3/EnterBrnn_classification/cell_1/bidirectional_rnn/bw/bw/while/Identity_1Qrnn_classification/cell_1/bidirectional_rnn/bw/bw/while/TensorArrayReadV3/Enter_1*
dtype0*
_output_shapes
:	
Ă
Ornn_classification/cell_1/bidirectional_rnn/bw/bw/while/TensorArrayReadV3/EnterEnter?rnn_classification/cell_1/bidirectional_rnn/bw/bw/TensorArray_1*
T0*
is_constant(*
parallel_iterations *U

frame_nameGErnn_classification/cell_1/bidirectional_rnn/bw/bw/while/while_context*
_output_shapes
:
î
Qrnn_classification/cell_1/bidirectional_rnn/bw/bw/while/TensorArrayReadV3/Enter_1Enterlrnn_classification/cell_1/bidirectional_rnn/bw/bw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3*
T0*
is_constant(*
parallel_iterations *U

frame_nameGErnn_classification/cell_1/bidirectional_rnn/bw/bw/while/while_context*
_output_shapes
: 

Drnn_classification/cell_1/bidirectional_rnn/bw/bw/while/GreaterEqualGreaterEqualBrnn_classification/cell_1/bidirectional_rnn/bw/bw/while/Identity_1Jrnn_classification/cell_1/bidirectional_rnn/bw/bw/while/GreaterEqual/Enter*
T0*
_output_shapes
:
ź
Jrnn_classification/cell_1/bidirectional_rnn/bw/bw/while/GreaterEqual/EnterEnter=rnn_classification/cell_1/bidirectional_rnn/bw/bw/CheckSeqLen*
parallel_iterations *U

frame_nameGErnn_classification/cell_1/bidirectional_rnn/bw/bw/while/while_context*
_output_shapes
:*
T0*
is_constant(

frnn_classification/cell_1/bidirectional_rnn/bw/basic_lstm_cell/kernel/Initializer/random_uniform/shapeConst*
valueB"     *X
_classN
LJloc:@rnn_classification/cell_1/bidirectional_rnn/bw/basic_lstm_cell/kernel*
dtype0*
_output_shapes
:

drnn_classification/cell_1/bidirectional_rnn/bw/basic_lstm_cell/kernel/Initializer/random_uniform/minConst*
valueB
 *b§˝*X
_classN
LJloc:@rnn_classification/cell_1/bidirectional_rnn/bw/basic_lstm_cell/kernel*
dtype0*
_output_shapes
: 

drnn_classification/cell_1/bidirectional_rnn/bw/basic_lstm_cell/kernel/Initializer/random_uniform/maxConst*
valueB
 *b§=*X
_classN
LJloc:@rnn_classification/cell_1/bidirectional_rnn/bw/basic_lstm_cell/kernel*
dtype0*
_output_shapes
: 
ů
nrnn_classification/cell_1/bidirectional_rnn/bw/basic_lstm_cell/kernel/Initializer/random_uniform/RandomUniformRandomUniformfrnn_classification/cell_1/bidirectional_rnn/bw/basic_lstm_cell/kernel/Initializer/random_uniform/shape*
T0*X
_classN
LJloc:@rnn_classification/cell_1/bidirectional_rnn/bw/basic_lstm_cell/kernel*
dtype0* 
_output_shapes
:

˛
drnn_classification/cell_1/bidirectional_rnn/bw/basic_lstm_cell/kernel/Initializer/random_uniform/subSubdrnn_classification/cell_1/bidirectional_rnn/bw/basic_lstm_cell/kernel/Initializer/random_uniform/maxdrnn_classification/cell_1/bidirectional_rnn/bw/basic_lstm_cell/kernel/Initializer/random_uniform/min*
T0*X
_classN
LJloc:@rnn_classification/cell_1/bidirectional_rnn/bw/basic_lstm_cell/kernel*
_output_shapes
: 
Ć
drnn_classification/cell_1/bidirectional_rnn/bw/basic_lstm_cell/kernel/Initializer/random_uniform/mulMulnrnn_classification/cell_1/bidirectional_rnn/bw/basic_lstm_cell/kernel/Initializer/random_uniform/RandomUniformdrnn_classification/cell_1/bidirectional_rnn/bw/basic_lstm_cell/kernel/Initializer/random_uniform/sub*
T0*X
_classN
LJloc:@rnn_classification/cell_1/bidirectional_rnn/bw/basic_lstm_cell/kernel* 
_output_shapes
:

¸
`rnn_classification/cell_1/bidirectional_rnn/bw/basic_lstm_cell/kernel/Initializer/random_uniformAdddrnn_classification/cell_1/bidirectional_rnn/bw/basic_lstm_cell/kernel/Initializer/random_uniform/muldrnn_classification/cell_1/bidirectional_rnn/bw/basic_lstm_cell/kernel/Initializer/random_uniform/min* 
_output_shapes
:
*
T0*X
_classN
LJloc:@rnn_classification/cell_1/bidirectional_rnn/bw/basic_lstm_cell/kernel
ó
Ernn_classification/cell_1/bidirectional_rnn/bw/basic_lstm_cell/kernel
VariableV2*
dtype0* 
_output_shapes
:
*X
_classN
LJloc:@rnn_classification/cell_1/bidirectional_rnn/bw/basic_lstm_cell/kernel*
shape:


Lrnn_classification/cell_1/bidirectional_rnn/bw/basic_lstm_cell/kernel/AssignAssignErnn_classification/cell_1/bidirectional_rnn/bw/basic_lstm_cell/kernel`rnn_classification/cell_1/bidirectional_rnn/bw/basic_lstm_cell/kernel/Initializer/random_uniform*
T0*X
_classN
LJloc:@rnn_classification/cell_1/bidirectional_rnn/bw/basic_lstm_cell/kernel* 
_output_shapes
:

Č
Jrnn_classification/cell_1/bidirectional_rnn/bw/basic_lstm_cell/kernel/readIdentityErnn_classification/cell_1/bidirectional_rnn/bw/basic_lstm_cell/kernel*
T0* 
_output_shapes
:

ü
Urnn_classification/cell_1/bidirectional_rnn/bw/basic_lstm_cell/bias/Initializer/zerosConst*
valueB*    *V
_classL
JHloc:@rnn_classification/cell_1/bidirectional_rnn/bw/basic_lstm_cell/bias*
dtype0*
_output_shapes	
:
ĺ
Crnn_classification/cell_1/bidirectional_rnn/bw/basic_lstm_cell/bias
VariableV2*
dtype0*
_output_shapes	
:*V
_classL
JHloc:@rnn_classification/cell_1/bidirectional_rnn/bw/basic_lstm_cell/bias*
shape:
î
Jrnn_classification/cell_1/bidirectional_rnn/bw/basic_lstm_cell/bias/AssignAssignCrnn_classification/cell_1/bidirectional_rnn/bw/basic_lstm_cell/biasUrnn_classification/cell_1/bidirectional_rnn/bw/basic_lstm_cell/bias/Initializer/zeros*
T0*V
_classL
JHloc:@rnn_classification/cell_1/bidirectional_rnn/bw/basic_lstm_cell/bias*
_output_shapes	
:
ż
Hrnn_classification/cell_1/bidirectional_rnn/bw/basic_lstm_cell/bias/readIdentityCrnn_classification/cell_1/bidirectional_rnn/bw/basic_lstm_cell/bias*
T0*
_output_shapes	
:
Ň
Mrnn_classification/cell_1/bidirectional_rnn/bw/bw/while/basic_lstm_cell/ConstConstA^rnn_classification/cell_1/bidirectional_rnn/bw/bw/while/Identity*
dtype0*
_output_shapes
: *
value	B :
Ř
Srnn_classification/cell_1/bidirectional_rnn/bw/bw/while/basic_lstm_cell/concat/axisConstA^rnn_classification/cell_1/bidirectional_rnn/bw/bw/while/Identity*
dtype0*
_output_shapes
: *
value	B :
ń
Nrnn_classification/cell_1/bidirectional_rnn/bw/bw/while/basic_lstm_cell/concatConcatV2Irnn_classification/cell_1/bidirectional_rnn/bw/bw/while/TensorArrayReadV3Brnn_classification/cell_1/bidirectional_rnn/bw/bw/while/Identity_4Srnn_classification/cell_1/bidirectional_rnn/bw/bw/while/basic_lstm_cell/concat/axis*
N*
_output_shapes
:	*
T0
¨
Nrnn_classification/cell_1/bidirectional_rnn/bw/bw/while/basic_lstm_cell/MatMulMatMulNrnn_classification/cell_1/bidirectional_rnn/bw/bw/while/basic_lstm_cell/concatTrnn_classification/cell_1/bidirectional_rnn/bw/bw/while/basic_lstm_cell/MatMul/Enter*
T0*
_output_shapes
:	
Ů
Trnn_classification/cell_1/bidirectional_rnn/bw/bw/while/basic_lstm_cell/MatMul/EnterEnterJrnn_classification/cell_1/bidirectional_rnn/bw/basic_lstm_cell/kernel/read*
T0*
is_constant(*
parallel_iterations *U

frame_nameGErnn_classification/cell_1/bidirectional_rnn/bw/bw/while/while_context* 
_output_shapes
:

Ť
Ornn_classification/cell_1/bidirectional_rnn/bw/bw/while/basic_lstm_cell/BiasAddBiasAddNrnn_classification/cell_1/bidirectional_rnn/bw/bw/while/basic_lstm_cell/MatMulUrnn_classification/cell_1/bidirectional_rnn/bw/bw/while/basic_lstm_cell/BiasAdd/Enter*
_output_shapes
:	*
T0
Ó
Urnn_classification/cell_1/bidirectional_rnn/bw/bw/while/basic_lstm_cell/BiasAdd/EnterEnterHrnn_classification/cell_1/bidirectional_rnn/bw/basic_lstm_cell/bias/read*
parallel_iterations *U

frame_nameGErnn_classification/cell_1/bidirectional_rnn/bw/bw/while/while_context*
_output_shapes	
:*
T0*
is_constant(
Ô
Ornn_classification/cell_1/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Const_1ConstA^rnn_classification/cell_1/bidirectional_rnn/bw/bw/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
Ň
Mrnn_classification/cell_1/bidirectional_rnn/bw/bw/while/basic_lstm_cell/splitSplitMrnn_classification/cell_1/bidirectional_rnn/bw/bw/while/basic_lstm_cell/ConstOrnn_classification/cell_1/bidirectional_rnn/bw/bw/while/basic_lstm_cell/BiasAdd*
T0*
	num_split*@
_output_shapes.
,:	:	:	:	
×
Ornn_classification/cell_1/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Const_2ConstA^rnn_classification/cell_1/bidirectional_rnn/bw/bw/while/Identity*
valueB
 *  ?*
dtype0*
_output_shapes
: 

Krnn_classification/cell_1/bidirectional_rnn/bw/bw/while/basic_lstm_cell/AddAddOrnn_classification/cell_1/bidirectional_rnn/bw/bw/while/basic_lstm_cell/split:2Ornn_classification/cell_1/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Const_2*
T0*
_output_shapes
:	
Ń
Ornn_classification/cell_1/bidirectional_rnn/bw/bw/while/basic_lstm_cell/SigmoidSigmoidKrnn_classification/cell_1/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Add*
T0*
_output_shapes
:	

Krnn_classification/cell_1/bidirectional_rnn/bw/bw/while/basic_lstm_cell/MulMulBrnn_classification/cell_1/bidirectional_rnn/bw/bw/while/Identity_3Ornn_classification/cell_1/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Sigmoid*
T0*
_output_shapes
:	
Ő
Qrnn_classification/cell_1/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Sigmoid_1SigmoidMrnn_classification/cell_1/bidirectional_rnn/bw/bw/while/basic_lstm_cell/split*
T0*
_output_shapes
:	
Ď
Lrnn_classification/cell_1/bidirectional_rnn/bw/bw/while/basic_lstm_cell/TanhTanhOrnn_classification/cell_1/bidirectional_rnn/bw/bw/while/basic_lstm_cell/split:1*
T0*
_output_shapes
:	

Mrnn_classification/cell_1/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_1MulQrnn_classification/cell_1/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Sigmoid_1Lrnn_classification/cell_1/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Tanh*
T0*
_output_shapes
:	

Mrnn_classification/cell_1/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Add_1AddKrnn_classification/cell_1/bidirectional_rnn/bw/bw/while/basic_lstm_cell/MulMrnn_classification/cell_1/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_1*
T0*
_output_shapes
:	
Ď
Nrnn_classification/cell_1/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Tanh_1TanhMrnn_classification/cell_1/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Add_1*
T0*
_output_shapes
:	
×
Qrnn_classification/cell_1/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Sigmoid_2SigmoidOrnn_classification/cell_1/bidirectional_rnn/bw/bw/while/basic_lstm_cell/split:3*
T0*
_output_shapes
:	
Ą
Mrnn_classification/cell_1/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_2MulNrnn_classification/cell_1/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Tanh_1Qrnn_classification/cell_1/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Sigmoid_2*
T0*
_output_shapes
:	
Ż
>rnn_classification/cell_1/bidirectional_rnn/bw/bw/while/SelectSelectDrnn_classification/cell_1/bidirectional_rnn/bw/bw/while/GreaterEqualDrnn_classification/cell_1/bidirectional_rnn/bw/bw/while/Select/EnterMrnn_classification/cell_1/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_2*
T0*`
_classV
TRloc:@rnn_classification/cell_1/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_2*
_output_shapes
:	

Drnn_classification/cell_1/bidirectional_rnn/bw/bw/while/Select/EnterEnter7rnn_classification/cell_1/bidirectional_rnn/bw/bw/zeros*
parallel_iterations *
is_constant(*
_output_shapes
:	*U

frame_nameGErnn_classification/cell_1/bidirectional_rnn/bw/bw/while/while_context*
T0*`
_classV
TRloc:@rnn_classification/cell_1/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_2
Ż
@rnn_classification/cell_1/bidirectional_rnn/bw/bw/while/Select_1SelectDrnn_classification/cell_1/bidirectional_rnn/bw/bw/while/GreaterEqualBrnn_classification/cell_1/bidirectional_rnn/bw/bw/while/Identity_3Mrnn_classification/cell_1/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Add_1*
T0*`
_classV
TRloc:@rnn_classification/cell_1/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Add_1*
_output_shapes
:	
Ż
@rnn_classification/cell_1/bidirectional_rnn/bw/bw/while/Select_2SelectDrnn_classification/cell_1/bidirectional_rnn/bw/bw/while/GreaterEqualBrnn_classification/cell_1/bidirectional_rnn/bw/bw/while/Identity_4Mrnn_classification/cell_1/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_2*
T0*`
_classV
TRloc:@rnn_classification/cell_1/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_2*
_output_shapes
:	

[rnn_classification/cell_1/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3TensorArrayWriteV3arnn_classification/cell_1/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3/EnterBrnn_classification/cell_1/bidirectional_rnn/bw/bw/while/Identity_1>rnn_classification/cell_1/bidirectional_rnn/bw/bw/while/SelectBrnn_classification/cell_1/bidirectional_rnn/bw/bw/while/Identity_2*
_output_shapes
: *
T0*`
_classV
TRloc:@rnn_classification/cell_1/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_2
ľ
arnn_classification/cell_1/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3/EnterEnter=rnn_classification/cell_1/bidirectional_rnn/bw/bw/TensorArray*
is_constant(*U

frame_nameGErnn_classification/cell_1/bidirectional_rnn/bw/bw/while/while_context*
_output_shapes
:*
T0*`
_classV
TRloc:@rnn_classification/cell_1/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_2*
parallel_iterations 
Ä
?rnn_classification/cell_1/bidirectional_rnn/bw/bw/while/add_1/yConstA^rnn_classification/cell_1/bidirectional_rnn/bw/bw/while/Identity*
dtype0*
_output_shapes
: *
value	B :
ę
=rnn_classification/cell_1/bidirectional_rnn/bw/bw/while/add_1AddBrnn_classification/cell_1/bidirectional_rnn/bw/bw/while/Identity_1?rnn_classification/cell_1/bidirectional_rnn/bw/bw/while/add_1/y*
T0*
_output_shapes
: 
´
Ernn_classification/cell_1/bidirectional_rnn/bw/bw/while/NextIterationNextIteration;rnn_classification/cell_1/bidirectional_rnn/bw/bw/while/add*
T0*
_output_shapes
: 
¸
Grnn_classification/cell_1/bidirectional_rnn/bw/bw/while/NextIteration_1NextIteration=rnn_classification/cell_1/bidirectional_rnn/bw/bw/while/add_1*
_output_shapes
: *
T0
Ö
Grnn_classification/cell_1/bidirectional_rnn/bw/bw/while/NextIteration_2NextIteration[rnn_classification/cell_1/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3*
T0*
_output_shapes
: 
Ä
Grnn_classification/cell_1/bidirectional_rnn/bw/bw/while/NextIteration_3NextIteration@rnn_classification/cell_1/bidirectional_rnn/bw/bw/while/Select_1*
T0*
_output_shapes
:	
Ä
Grnn_classification/cell_1/bidirectional_rnn/bw/bw/while/NextIteration_4NextIteration@rnn_classification/cell_1/bidirectional_rnn/bw/bw/while/Select_2*
T0*
_output_shapes
:	
Ľ
<rnn_classification/cell_1/bidirectional_rnn/bw/bw/while/ExitExit>rnn_classification/cell_1/bidirectional_rnn/bw/bw/while/Switch*
_output_shapes
: *
T0
Š
>rnn_classification/cell_1/bidirectional_rnn/bw/bw/while/Exit_1Exit@rnn_classification/cell_1/bidirectional_rnn/bw/bw/while/Switch_1*
_output_shapes
: *
T0
Š
>rnn_classification/cell_1/bidirectional_rnn/bw/bw/while/Exit_2Exit@rnn_classification/cell_1/bidirectional_rnn/bw/bw/while/Switch_2*
_output_shapes
: *
T0
˛
>rnn_classification/cell_1/bidirectional_rnn/bw/bw/while/Exit_3Exit@rnn_classification/cell_1/bidirectional_rnn/bw/bw/while/Switch_3*
T0*
_output_shapes
:	
˛
>rnn_classification/cell_1/bidirectional_rnn/bw/bw/while/Exit_4Exit@rnn_classification/cell_1/bidirectional_rnn/bw/bw/while/Switch_4*
T0*
_output_shapes
:	
Ň
Trnn_classification/cell_1/bidirectional_rnn/bw/bw/TensorArrayStack/TensorArraySizeV3TensorArraySizeV3=rnn_classification/cell_1/bidirectional_rnn/bw/bw/TensorArray>rnn_classification/cell_1/bidirectional_rnn/bw/bw/while/Exit_2*P
_classF
DBloc:@rnn_classification/cell_1/bidirectional_rnn/bw/bw/TensorArray*
_output_shapes
: 
â
Nrnn_classification/cell_1/bidirectional_rnn/bw/bw/TensorArrayStack/range/startConst*
value	B : *P
_classF
DBloc:@rnn_classification/cell_1/bidirectional_rnn/bw/bw/TensorArray*
dtype0*
_output_shapes
: 
â
Nrnn_classification/cell_1/bidirectional_rnn/bw/bw/TensorArrayStack/range/deltaConst*
value	B :*P
_classF
DBloc:@rnn_classification/cell_1/bidirectional_rnn/bw/bw/TensorArray*
dtype0*
_output_shapes
: 
ž
Hrnn_classification/cell_1/bidirectional_rnn/bw/bw/TensorArrayStack/rangeRangeNrnn_classification/cell_1/bidirectional_rnn/bw/bw/TensorArrayStack/range/startTrnn_classification/cell_1/bidirectional_rnn/bw/bw/TensorArrayStack/TensorArraySizeV3Nrnn_classification/cell_1/bidirectional_rnn/bw/bw/TensorArrayStack/range/delta*P
_classF
DBloc:@rnn_classification/cell_1/bidirectional_rnn/bw/bw/TensorArray*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
á
Vrnn_classification/cell_1/bidirectional_rnn/bw/bw/TensorArrayStack/TensorArrayGatherV3TensorArrayGatherV3=rnn_classification/cell_1/bidirectional_rnn/bw/bw/TensorArrayHrnn_classification/cell_1/bidirectional_rnn/bw/bw/TensorArrayStack/range>rnn_classification/cell_1/bidirectional_rnn/bw/bw/while/Exit_2*P
_classF
DBloc:@rnn_classification/cell_1/bidirectional_rnn/bw/bw/TensorArray*
dtype0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙*
element_shape:	

9rnn_classification/cell_1/bidirectional_rnn/bw/bw/Const_5Const*
dtype0*
_output_shapes
:*
valueB:
z
8rnn_classification/cell_1/bidirectional_rnn/bw/bw/Rank_1Const*
value	B :*
dtype0*
_output_shapes
: 

?rnn_classification/cell_1/bidirectional_rnn/bw/bw/range_1/startConst*
value	B :*
dtype0*
_output_shapes
: 

?rnn_classification/cell_1/bidirectional_rnn/bw/bw/range_1/deltaConst*
value	B :*
dtype0*
_output_shapes
: 

9rnn_classification/cell_1/bidirectional_rnn/bw/bw/range_1Range?rnn_classification/cell_1/bidirectional_rnn/bw/bw/range_1/start8rnn_classification/cell_1/bidirectional_rnn/bw/bw/Rank_1?rnn_classification/cell_1/bidirectional_rnn/bw/bw/range_1/delta*
_output_shapes
:

Crnn_classification/cell_1/bidirectional_rnn/bw/bw/concat_2/values_0Const*
valueB"       *
dtype0*
_output_shapes
:

?rnn_classification/cell_1/bidirectional_rnn/bw/bw/concat_2/axisConst*
value	B : *
dtype0*
_output_shapes
: 
ľ
:rnn_classification/cell_1/bidirectional_rnn/bw/bw/concat_2ConcatV2Crnn_classification/cell_1/bidirectional_rnn/bw/bw/concat_2/values_09rnn_classification/cell_1/bidirectional_rnn/bw/bw/range_1?rnn_classification/cell_1/bidirectional_rnn/bw/bw/concat_2/axis*
N*
_output_shapes
:*
T0

=rnn_classification/cell_1/bidirectional_rnn/bw/bw/transpose_1	TransposeVrnn_classification/cell_1/bidirectional_rnn/bw/bw/TensorArrayStack/TensorArrayGatherV3:rnn_classification/cell_1/bidirectional_rnn/bw/bw/concat_2*,
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ę
)rnn_classification/cell_1/ReverseSequenceReverseSequence=rnn_classification/cell_1/bidirectional_rnn/bw/bw/transpose_1Squeeze*
seq_dim*,
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
g
%rnn_classification/cell_1/concat/axisConst*
dtype0*
_output_shapes
: *
value	B :
ý
 rnn_classification/cell_1/concatConcatV2=rnn_classification/cell_1/bidirectional_rnn/fw/fw/transpose_1)rnn_classification/cell_1/ReverseSequence%rnn_classification/cell_1/concat/axis*
T0*
N*,
_output_shapes
:˙˙˙˙˙˙˙˙˙
x
6rnn_classification/cell_2/bidirectional_rnn/fw/fw/RankConst*
dtype0*
_output_shapes
: *
value	B :

=rnn_classification/cell_2/bidirectional_rnn/fw/fw/range/startConst*
value	B :*
dtype0*
_output_shapes
: 

=rnn_classification/cell_2/bidirectional_rnn/fw/fw/range/deltaConst*
dtype0*
_output_shapes
: *
value	B :

7rnn_classification/cell_2/bidirectional_rnn/fw/fw/rangeRange=rnn_classification/cell_2/bidirectional_rnn/fw/fw/range/start6rnn_classification/cell_2/bidirectional_rnn/fw/fw/Rank=rnn_classification/cell_2/bidirectional_rnn/fw/fw/range/delta*
_output_shapes
:

Arnn_classification/cell_2/bidirectional_rnn/fw/fw/concat/values_0Const*
valueB"       *
dtype0*
_output_shapes
:

=rnn_classification/cell_2/bidirectional_rnn/fw/fw/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
­
8rnn_classification/cell_2/bidirectional_rnn/fw/fw/concatConcatV2Arnn_classification/cell_2/bidirectional_rnn/fw/fw/concat/values_07rnn_classification/cell_2/bidirectional_rnn/fw/fw/range=rnn_classification/cell_2/bidirectional_rnn/fw/fw/concat/axis*
N*
_output_shapes
:*
T0
Ű
;rnn_classification/cell_2/bidirectional_rnn/fw/fw/transpose	Transpose rnn_classification/cell_1/concat8rnn_classification/cell_2/bidirectional_rnn/fw/fw/concat*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙
~
9rnn_classification/cell_2/bidirectional_rnn/fw/fw/ToInt32CastSqueeze*

SrcT0	*

DstT0*
_output_shapes
:
­
Arnn_classification/cell_2/bidirectional_rnn/fw/fw/sequence_lengthIdentity9rnn_classification/cell_2/bidirectional_rnn/fw/fw/ToInt32*
_output_shapes
:*
T0
°
frnn_classification/cell_2/bidirectional_rnn/fw/fw/DropoutWrapperZeroState/BasicLSTMCellZeroState/ConstConst*
valueB:*
dtype0*
_output_shapes
:
ł
hrnn_classification/cell_2/bidirectional_rnn/fw/fw/DropoutWrapperZeroState/BasicLSTMCellZeroState/Const_1Const*
dtype0*
_output_shapes
:*
valueB:
Ž
lrnn_classification/cell_2/bidirectional_rnn/fw/fw/DropoutWrapperZeroState/BasicLSTMCellZeroState/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
á
grnn_classification/cell_2/bidirectional_rnn/fw/fw/DropoutWrapperZeroState/BasicLSTMCellZeroState/concatConcatV2frnn_classification/cell_2/bidirectional_rnn/fw/fw/DropoutWrapperZeroState/BasicLSTMCellZeroState/Consthrnn_classification/cell_2/bidirectional_rnn/fw/fw/DropoutWrapperZeroState/BasicLSTMCellZeroState/Const_1lrnn_classification/cell_2/bidirectional_rnn/fw/fw/DropoutWrapperZeroState/BasicLSTMCellZeroState/concat/axis*
T0*
N*
_output_shapes
:
ą
lrnn_classification/cell_2/bidirectional_rnn/fw/fw/DropoutWrapperZeroState/BasicLSTMCellZeroState/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
ď
frnn_classification/cell_2/bidirectional_rnn/fw/fw/DropoutWrapperZeroState/BasicLSTMCellZeroState/zerosFillgrnn_classification/cell_2/bidirectional_rnn/fw/fw/DropoutWrapperZeroState/BasicLSTMCellZeroState/concatlrnn_classification/cell_2/bidirectional_rnn/fw/fw/DropoutWrapperZeroState/BasicLSTMCellZeroState/zeros/Const*
T0*
_output_shapes
:	
˛
hrnn_classification/cell_2/bidirectional_rnn/fw/fw/DropoutWrapperZeroState/BasicLSTMCellZeroState/Const_2Const*
valueB:*
dtype0*
_output_shapes
:
ł
hrnn_classification/cell_2/bidirectional_rnn/fw/fw/DropoutWrapperZeroState/BasicLSTMCellZeroState/Const_3Const*
valueB:*
dtype0*
_output_shapes
:
˛
hrnn_classification/cell_2/bidirectional_rnn/fw/fw/DropoutWrapperZeroState/BasicLSTMCellZeroState/Const_4Const*
dtype0*
_output_shapes
:*
valueB:
ł
hrnn_classification/cell_2/bidirectional_rnn/fw/fw/DropoutWrapperZeroState/BasicLSTMCellZeroState/Const_5Const*
valueB:*
dtype0*
_output_shapes
:
°
nrnn_classification/cell_2/bidirectional_rnn/fw/fw/DropoutWrapperZeroState/BasicLSTMCellZeroState/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
ç
irnn_classification/cell_2/bidirectional_rnn/fw/fw/DropoutWrapperZeroState/BasicLSTMCellZeroState/concat_1ConcatV2hrnn_classification/cell_2/bidirectional_rnn/fw/fw/DropoutWrapperZeroState/BasicLSTMCellZeroState/Const_4hrnn_classification/cell_2/bidirectional_rnn/fw/fw/DropoutWrapperZeroState/BasicLSTMCellZeroState/Const_5nrnn_classification/cell_2/bidirectional_rnn/fw/fw/DropoutWrapperZeroState/BasicLSTMCellZeroState/concat_1/axis*
T0*
N*
_output_shapes
:
ł
nrnn_classification/cell_2/bidirectional_rnn/fw/fw/DropoutWrapperZeroState/BasicLSTMCellZeroState/zeros_1/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
ő
hrnn_classification/cell_2/bidirectional_rnn/fw/fw/DropoutWrapperZeroState/BasicLSTMCellZeroState/zeros_1Fillirnn_classification/cell_2/bidirectional_rnn/fw/fw/DropoutWrapperZeroState/BasicLSTMCellZeroState/concat_1nrnn_classification/cell_2/bidirectional_rnn/fw/fw/DropoutWrapperZeroState/BasicLSTMCellZeroState/zeros_1/Const*
_output_shapes
:	*
T0
˛
hrnn_classification/cell_2/bidirectional_rnn/fw/fw/DropoutWrapperZeroState/BasicLSTMCellZeroState/Const_6Const*
valueB:*
dtype0*
_output_shapes
:
ł
hrnn_classification/cell_2/bidirectional_rnn/fw/fw/DropoutWrapperZeroState/BasicLSTMCellZeroState/Const_7Const*
dtype0*
_output_shapes
:*
valueB:

7rnn_classification/cell_2/bidirectional_rnn/fw/fw/ShapeConst*
valueB:*
dtype0*
_output_shapes
:

7rnn_classification/cell_2/bidirectional_rnn/fw/fw/stackConst*
dtype0*
_output_shapes
:*
valueB:
×
7rnn_classification/cell_2/bidirectional_rnn/fw/fw/EqualEqual7rnn_classification/cell_2/bidirectional_rnn/fw/fw/Shape7rnn_classification/cell_2/bidirectional_rnn/fw/fw/stack*
T0*
_output_shapes
:

7rnn_classification/cell_2/bidirectional_rnn/fw/fw/ConstConst*
valueB: *
dtype0*
_output_shapes
:
Ć
5rnn_classification/cell_2/bidirectional_rnn/fw/fw/AllAll7rnn_classification/cell_2/bidirectional_rnn/fw/fw/Equal7rnn_classification/cell_2/bidirectional_rnn/fw/fw/Const*
_output_shapes
: 
ŕ
>rnn_classification/cell_2/bidirectional_rnn/fw/fw/Assert/ConstConst*
dtype0*
_output_shapes
: *r
valueiBg BaExpected shape for Tensor rnn_classification/cell_2/bidirectional_rnn/fw/fw/sequence_length:0 is 

@rnn_classification/cell_2/bidirectional_rnn/fw/fw/Assert/Const_1Const*
dtype0*
_output_shapes
: *!
valueB B but saw shape: 
č
Frnn_classification/cell_2/bidirectional_rnn/fw/fw/Assert/Assert/data_0Const*r
valueiBg BaExpected shape for Tensor rnn_classification/cell_2/bidirectional_rnn/fw/fw/sequence_length:0 is *
dtype0*
_output_shapes
: 

Frnn_classification/cell_2/bidirectional_rnn/fw/fw/Assert/Assert/data_2Const*
dtype0*
_output_shapes
: *!
valueB B but saw shape: 

?rnn_classification/cell_2/bidirectional_rnn/fw/fw/Assert/AssertAssert5rnn_classification/cell_2/bidirectional_rnn/fw/fw/AllFrnn_classification/cell_2/bidirectional_rnn/fw/fw/Assert/Assert/data_07rnn_classification/cell_2/bidirectional_rnn/fw/fw/stackFrnn_classification/cell_2/bidirectional_rnn/fw/fw/Assert/Assert/data_27rnn_classification/cell_2/bidirectional_rnn/fw/fw/Shape*
T
2
ó
=rnn_classification/cell_2/bidirectional_rnn/fw/fw/CheckSeqLenIdentityArnn_classification/cell_2/bidirectional_rnn/fw/fw/sequence_length@^rnn_classification/cell_2/bidirectional_rnn/fw/fw/Assert/Assert*
T0*
_output_shapes
:
¤
9rnn_classification/cell_2/bidirectional_rnn/fw/fw/Shape_1Shape;rnn_classification/cell_2/bidirectional_rnn/fw/fw/transpose*
T0*
_output_shapes
:

Ernn_classification/cell_2/bidirectional_rnn/fw/fw/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:

Grnn_classification/cell_2/bidirectional_rnn/fw/fw/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:

Grnn_classification/cell_2/bidirectional_rnn/fw/fw/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
Š
?rnn_classification/cell_2/bidirectional_rnn/fw/fw/strided_sliceStridedSlice9rnn_classification/cell_2/bidirectional_rnn/fw/fw/Shape_1Ernn_classification/cell_2/bidirectional_rnn/fw/fw/strided_slice/stackGrnn_classification/cell_2/bidirectional_rnn/fw/fw/strided_slice/stack_1Grnn_classification/cell_2/bidirectional_rnn/fw/fw/strided_slice/stack_2*
shrink_axis_mask*
_output_shapes
: *
Index0*
T0

9rnn_classification/cell_2/bidirectional_rnn/fw/fw/Const_1Const*
valueB:*
dtype0*
_output_shapes
:

9rnn_classification/cell_2/bidirectional_rnn/fw/fw/Const_2Const*
valueB:*
dtype0*
_output_shapes
:

?rnn_classification/cell_2/bidirectional_rnn/fw/fw/concat_1/axisConst*
dtype0*
_output_shapes
: *
value	B : 
Ť
:rnn_classification/cell_2/bidirectional_rnn/fw/fw/concat_1ConcatV29rnn_classification/cell_2/bidirectional_rnn/fw/fw/Const_19rnn_classification/cell_2/bidirectional_rnn/fw/fw/Const_2?rnn_classification/cell_2/bidirectional_rnn/fw/fw/concat_1/axis*
T0*
N*
_output_shapes
:

=rnn_classification/cell_2/bidirectional_rnn/fw/fw/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
ä
7rnn_classification/cell_2/bidirectional_rnn/fw/fw/zerosFill:rnn_classification/cell_2/bidirectional_rnn/fw/fw/concat_1=rnn_classification/cell_2/bidirectional_rnn/fw/fw/zeros/Const*
T0*
_output_shapes
:	

9rnn_classification/cell_2/bidirectional_rnn/fw/fw/Const_3Const*
valueB: *
dtype0*
_output_shapes
:
×
5rnn_classification/cell_2/bidirectional_rnn/fw/fw/MinMin=rnn_classification/cell_2/bidirectional_rnn/fw/fw/CheckSeqLen9rnn_classification/cell_2/bidirectional_rnn/fw/fw/Const_3*
T0*
_output_shapes
: 

9rnn_classification/cell_2/bidirectional_rnn/fw/fw/Const_4Const*
valueB: *
dtype0*
_output_shapes
:
×
5rnn_classification/cell_2/bidirectional_rnn/fw/fw/MaxMax=rnn_classification/cell_2/bidirectional_rnn/fw/fw/CheckSeqLen9rnn_classification/cell_2/bidirectional_rnn/fw/fw/Const_4*
T0*
_output_shapes
: 
x
6rnn_classification/cell_2/bidirectional_rnn/fw/fw/timeConst*
value	B : *
dtype0*
_output_shapes
: 
×
=rnn_classification/cell_2/bidirectional_rnn/fw/fw/TensorArrayTensorArrayV3?rnn_classification/cell_2/bidirectional_rnn/fw/fw/strided_slice*]
tensor_array_nameHFrnn_classification/cell_2/bidirectional_rnn/fw/fw/dynamic_rnn/output_0*
dtype0*
_output_shapes

:: *
element_shape:	*
identical_element_shapes(
Ř
?rnn_classification/cell_2/bidirectional_rnn/fw/fw/TensorArray_1TensorArrayV3?rnn_classification/cell_2/bidirectional_rnn/fw/fw/strided_slice*
identical_element_shapes(*\
tensor_array_nameGErnn_classification/cell_2/bidirectional_rnn/fw/fw/dynamic_rnn/input_0*
dtype0*
_output_shapes

:: *
element_shape:	
ľ
Jrnn_classification/cell_2/bidirectional_rnn/fw/fw/TensorArrayUnstack/ShapeShape;rnn_classification/cell_2/bidirectional_rnn/fw/fw/transpose*
_output_shapes
:*
T0
˘
Xrnn_classification/cell_2/bidirectional_rnn/fw/fw/TensorArrayUnstack/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
¤
Zrnn_classification/cell_2/bidirectional_rnn/fw/fw/TensorArrayUnstack/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
¤
Zrnn_classification/cell_2/bidirectional_rnn/fw/fw/TensorArrayUnstack/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:

Rrnn_classification/cell_2/bidirectional_rnn/fw/fw/TensorArrayUnstack/strided_sliceStridedSliceJrnn_classification/cell_2/bidirectional_rnn/fw/fw/TensorArrayUnstack/ShapeXrnn_classification/cell_2/bidirectional_rnn/fw/fw/TensorArrayUnstack/strided_slice/stackZrnn_classification/cell_2/bidirectional_rnn/fw/fw/TensorArrayUnstack/strided_slice/stack_1Zrnn_classification/cell_2/bidirectional_rnn/fw/fw/TensorArrayUnstack/strided_slice/stack_2*
T0*
Index0*
shrink_axis_mask*
_output_shapes
: 

Prnn_classification/cell_2/bidirectional_rnn/fw/fw/TensorArrayUnstack/range/startConst*
value	B : *
dtype0*
_output_shapes
: 

Prnn_classification/cell_2/bidirectional_rnn/fw/fw/TensorArrayUnstack/range/deltaConst*
dtype0*
_output_shapes
: *
value	B :
đ
Jrnn_classification/cell_2/bidirectional_rnn/fw/fw/TensorArrayUnstack/rangeRangePrnn_classification/cell_2/bidirectional_rnn/fw/fw/TensorArrayUnstack/range/startRrnn_classification/cell_2/bidirectional_rnn/fw/fw/TensorArrayUnstack/strided_slicePrnn_classification/cell_2/bidirectional_rnn/fw/fw/TensorArrayUnstack/range/delta*#
_output_shapes
:˙˙˙˙˙˙˙˙˙

lrnn_classification/cell_2/bidirectional_rnn/fw/fw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3TensorArrayScatterV3?rnn_classification/cell_2/bidirectional_rnn/fw/fw/TensorArray_1Jrnn_classification/cell_2/bidirectional_rnn/fw/fw/TensorArrayUnstack/range;rnn_classification/cell_2/bidirectional_rnn/fw/fw/transposeArnn_classification/cell_2/bidirectional_rnn/fw/fw/TensorArray_1:1*
T0*N
_classD
B@loc:@rnn_classification/cell_2/bidirectional_rnn/fw/fw/transpose*
_output_shapes
: 
}
;rnn_classification/cell_2/bidirectional_rnn/fw/fw/Maximum/xConst*
value	B :*
dtype0*
_output_shapes
: 
Ů
9rnn_classification/cell_2/bidirectional_rnn/fw/fw/MaximumMaximum;rnn_classification/cell_2/bidirectional_rnn/fw/fw/Maximum/x5rnn_classification/cell_2/bidirectional_rnn/fw/fw/Max*
T0*
_output_shapes
: 
á
9rnn_classification/cell_2/bidirectional_rnn/fw/fw/MinimumMinimum?rnn_classification/cell_2/bidirectional_rnn/fw/fw/strided_slice9rnn_classification/cell_2/bidirectional_rnn/fw/fw/Maximum*
_output_shapes
: *
T0

Irnn_classification/cell_2/bidirectional_rnn/fw/fw/while/iteration_counterConst*
dtype0*
_output_shapes
: *
value	B : 
¤
=rnn_classification/cell_2/bidirectional_rnn/fw/fw/while/EnterEnterIrnn_classification/cell_2/bidirectional_rnn/fw/fw/while/iteration_counter*
T0*
parallel_iterations *U

frame_nameGErnn_classification/cell_2/bidirectional_rnn/fw/fw/while/while_context*
_output_shapes
: 

?rnn_classification/cell_2/bidirectional_rnn/fw/fw/while/Enter_1Enter6rnn_classification/cell_2/bidirectional_rnn/fw/fw/time*
T0*
parallel_iterations *U

frame_nameGErnn_classification/cell_2/bidirectional_rnn/fw/fw/while/while_context*
_output_shapes
: 

?rnn_classification/cell_2/bidirectional_rnn/fw/fw/while/Enter_2Enter?rnn_classification/cell_2/bidirectional_rnn/fw/fw/TensorArray:1*
T0*
parallel_iterations *U

frame_nameGErnn_classification/cell_2/bidirectional_rnn/fw/fw/while/while_context*
_output_shapes
: 
Ě
?rnn_classification/cell_2/bidirectional_rnn/fw/fw/while/Enter_3Enterfrnn_classification/cell_2/bidirectional_rnn/fw/fw/DropoutWrapperZeroState/BasicLSTMCellZeroState/zeros*
T0*
parallel_iterations *U

frame_nameGErnn_classification/cell_2/bidirectional_rnn/fw/fw/while/while_context*
_output_shapes
:	
Î
?rnn_classification/cell_2/bidirectional_rnn/fw/fw/while/Enter_4Enterhrnn_classification/cell_2/bidirectional_rnn/fw/fw/DropoutWrapperZeroState/BasicLSTMCellZeroState/zeros_1*
T0*
parallel_iterations *U

frame_nameGErnn_classification/cell_2/bidirectional_rnn/fw/fw/while/while_context*
_output_shapes
:	
ř
=rnn_classification/cell_2/bidirectional_rnn/fw/fw/while/MergeMerge=rnn_classification/cell_2/bidirectional_rnn/fw/fw/while/EnterErnn_classification/cell_2/bidirectional_rnn/fw/fw/while/NextIteration*
T0*
N*
_output_shapes
: : 
ţ
?rnn_classification/cell_2/bidirectional_rnn/fw/fw/while/Merge_1Merge?rnn_classification/cell_2/bidirectional_rnn/fw/fw/while/Enter_1Grnn_classification/cell_2/bidirectional_rnn/fw/fw/while/NextIteration_1*
N*
_output_shapes
: : *
T0
ţ
?rnn_classification/cell_2/bidirectional_rnn/fw/fw/while/Merge_2Merge?rnn_classification/cell_2/bidirectional_rnn/fw/fw/while/Enter_2Grnn_classification/cell_2/bidirectional_rnn/fw/fw/while/NextIteration_2*
T0*
N*
_output_shapes
: : 

?rnn_classification/cell_2/bidirectional_rnn/fw/fw/while/Merge_3Merge?rnn_classification/cell_2/bidirectional_rnn/fw/fw/while/Enter_3Grnn_classification/cell_2/bidirectional_rnn/fw/fw/while/NextIteration_3*
T0*
N*!
_output_shapes
:	: 

?rnn_classification/cell_2/bidirectional_rnn/fw/fw/while/Merge_4Merge?rnn_classification/cell_2/bidirectional_rnn/fw/fw/while/Enter_4Grnn_classification/cell_2/bidirectional_rnn/fw/fw/while/NextIteration_4*
N*!
_output_shapes
:	: *
T0
č
<rnn_classification/cell_2/bidirectional_rnn/fw/fw/while/LessLess=rnn_classification/cell_2/bidirectional_rnn/fw/fw/while/MergeBrnn_classification/cell_2/bidirectional_rnn/fw/fw/while/Less/Enter*
T0*
_output_shapes
: 
˛
Brnn_classification/cell_2/bidirectional_rnn/fw/fw/while/Less/EnterEnter?rnn_classification/cell_2/bidirectional_rnn/fw/fw/strided_slice*
T0*
is_constant(*
parallel_iterations *U

frame_nameGErnn_classification/cell_2/bidirectional_rnn/fw/fw/while/while_context*
_output_shapes
: 
î
>rnn_classification/cell_2/bidirectional_rnn/fw/fw/while/Less_1Less?rnn_classification/cell_2/bidirectional_rnn/fw/fw/while/Merge_1Drnn_classification/cell_2/bidirectional_rnn/fw/fw/while/Less_1/Enter*
T0*
_output_shapes
: 
Ž
Drnn_classification/cell_2/bidirectional_rnn/fw/fw/while/Less_1/EnterEnter9rnn_classification/cell_2/bidirectional_rnn/fw/fw/Minimum*
T0*
is_constant(*
parallel_iterations *U

frame_nameGErnn_classification/cell_2/bidirectional_rnn/fw/fw/while/while_context*
_output_shapes
: 
ć
Brnn_classification/cell_2/bidirectional_rnn/fw/fw/while/LogicalAnd
LogicalAnd<rnn_classification/cell_2/bidirectional_rnn/fw/fw/while/Less>rnn_classification/cell_2/bidirectional_rnn/fw/fw/while/Less_1*
_output_shapes
: 
¨
@rnn_classification/cell_2/bidirectional_rnn/fw/fw/while/LoopCondLoopCondBrnn_classification/cell_2/bidirectional_rnn/fw/fw/while/LogicalAnd*
_output_shapes
: 
ž
>rnn_classification/cell_2/bidirectional_rnn/fw/fw/while/SwitchSwitch=rnn_classification/cell_2/bidirectional_rnn/fw/fw/while/Merge@rnn_classification/cell_2/bidirectional_rnn/fw/fw/while/LoopCond*
T0*P
_classF
DBloc:@rnn_classification/cell_2/bidirectional_rnn/fw/fw/while/Merge*
_output_shapes
: : 
Ä
@rnn_classification/cell_2/bidirectional_rnn/fw/fw/while/Switch_1Switch?rnn_classification/cell_2/bidirectional_rnn/fw/fw/while/Merge_1@rnn_classification/cell_2/bidirectional_rnn/fw/fw/while/LoopCond*
T0*R
_classH
FDloc:@rnn_classification/cell_2/bidirectional_rnn/fw/fw/while/Merge_1*
_output_shapes
: : 
Ä
@rnn_classification/cell_2/bidirectional_rnn/fw/fw/while/Switch_2Switch?rnn_classification/cell_2/bidirectional_rnn/fw/fw/while/Merge_2@rnn_classification/cell_2/bidirectional_rnn/fw/fw/while/LoopCond*
T0*R
_classH
FDloc:@rnn_classification/cell_2/bidirectional_rnn/fw/fw/while/Merge_2*
_output_shapes
: : 
Ö
@rnn_classification/cell_2/bidirectional_rnn/fw/fw/while/Switch_3Switch?rnn_classification/cell_2/bidirectional_rnn/fw/fw/while/Merge_3@rnn_classification/cell_2/bidirectional_rnn/fw/fw/while/LoopCond*
T0*R
_classH
FDloc:@rnn_classification/cell_2/bidirectional_rnn/fw/fw/while/Merge_3**
_output_shapes
:	:	
Ö
@rnn_classification/cell_2/bidirectional_rnn/fw/fw/while/Switch_4Switch?rnn_classification/cell_2/bidirectional_rnn/fw/fw/while/Merge_4@rnn_classification/cell_2/bidirectional_rnn/fw/fw/while/LoopCond*
T0*R
_classH
FDloc:@rnn_classification/cell_2/bidirectional_rnn/fw/fw/while/Merge_4**
_output_shapes
:	:	
Ż
@rnn_classification/cell_2/bidirectional_rnn/fw/fw/while/IdentityIdentity@rnn_classification/cell_2/bidirectional_rnn/fw/fw/while/Switch:1*
_output_shapes
: *
T0
ł
Brnn_classification/cell_2/bidirectional_rnn/fw/fw/while/Identity_1IdentityBrnn_classification/cell_2/bidirectional_rnn/fw/fw/while/Switch_1:1*
T0*
_output_shapes
: 
ł
Brnn_classification/cell_2/bidirectional_rnn/fw/fw/while/Identity_2IdentityBrnn_classification/cell_2/bidirectional_rnn/fw/fw/while/Switch_2:1*
T0*
_output_shapes
: 
ź
Brnn_classification/cell_2/bidirectional_rnn/fw/fw/while/Identity_3IdentityBrnn_classification/cell_2/bidirectional_rnn/fw/fw/while/Switch_3:1*
_output_shapes
:	*
T0
ź
Brnn_classification/cell_2/bidirectional_rnn/fw/fw/while/Identity_4IdentityBrnn_classification/cell_2/bidirectional_rnn/fw/fw/while/Switch_4:1*
T0*
_output_shapes
:	
Â
=rnn_classification/cell_2/bidirectional_rnn/fw/fw/while/add/yConstA^rnn_classification/cell_2/bidirectional_rnn/fw/fw/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
ä
;rnn_classification/cell_2/bidirectional_rnn/fw/fw/while/addAdd@rnn_classification/cell_2/bidirectional_rnn/fw/fw/while/Identity=rnn_classification/cell_2/bidirectional_rnn/fw/fw/while/add/y*
T0*
_output_shapes
: 
ô
Irnn_classification/cell_2/bidirectional_rnn/fw/fw/while/TensorArrayReadV3TensorArrayReadV3Ornn_classification/cell_2/bidirectional_rnn/fw/fw/while/TensorArrayReadV3/EnterBrnn_classification/cell_2/bidirectional_rnn/fw/fw/while/Identity_1Qrnn_classification/cell_2/bidirectional_rnn/fw/fw/while/TensorArrayReadV3/Enter_1*
dtype0*
_output_shapes
:	
Ă
Ornn_classification/cell_2/bidirectional_rnn/fw/fw/while/TensorArrayReadV3/EnterEnter?rnn_classification/cell_2/bidirectional_rnn/fw/fw/TensorArray_1*
T0*
is_constant(*
parallel_iterations *U

frame_nameGErnn_classification/cell_2/bidirectional_rnn/fw/fw/while/while_context*
_output_shapes
:
î
Qrnn_classification/cell_2/bidirectional_rnn/fw/fw/while/TensorArrayReadV3/Enter_1Enterlrnn_classification/cell_2/bidirectional_rnn/fw/fw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3*
T0*
is_constant(*
parallel_iterations *U

frame_nameGErnn_classification/cell_2/bidirectional_rnn/fw/fw/while/while_context*
_output_shapes
: 

Drnn_classification/cell_2/bidirectional_rnn/fw/fw/while/GreaterEqualGreaterEqualBrnn_classification/cell_2/bidirectional_rnn/fw/fw/while/Identity_1Jrnn_classification/cell_2/bidirectional_rnn/fw/fw/while/GreaterEqual/Enter*
T0*
_output_shapes
:
ź
Jrnn_classification/cell_2/bidirectional_rnn/fw/fw/while/GreaterEqual/EnterEnter=rnn_classification/cell_2/bidirectional_rnn/fw/fw/CheckSeqLen*
T0*
is_constant(*
parallel_iterations *U

frame_nameGErnn_classification/cell_2/bidirectional_rnn/fw/fw/while/while_context*
_output_shapes
:

frnn_classification/cell_2/bidirectional_rnn/fw/basic_lstm_cell/kernel/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*
valueB"     *X
_classN
LJloc:@rnn_classification/cell_2/bidirectional_rnn/fw/basic_lstm_cell/kernel

drnn_classification/cell_2/bidirectional_rnn/fw/basic_lstm_cell/kernel/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *
valueB
 *b§˝*X
_classN
LJloc:@rnn_classification/cell_2/bidirectional_rnn/fw/basic_lstm_cell/kernel

drnn_classification/cell_2/bidirectional_rnn/fw/basic_lstm_cell/kernel/Initializer/random_uniform/maxConst*
valueB
 *b§=*X
_classN
LJloc:@rnn_classification/cell_2/bidirectional_rnn/fw/basic_lstm_cell/kernel*
dtype0*
_output_shapes
: 
ů
nrnn_classification/cell_2/bidirectional_rnn/fw/basic_lstm_cell/kernel/Initializer/random_uniform/RandomUniformRandomUniformfrnn_classification/cell_2/bidirectional_rnn/fw/basic_lstm_cell/kernel/Initializer/random_uniform/shape*
dtype0* 
_output_shapes
:
*
T0*X
_classN
LJloc:@rnn_classification/cell_2/bidirectional_rnn/fw/basic_lstm_cell/kernel
˛
drnn_classification/cell_2/bidirectional_rnn/fw/basic_lstm_cell/kernel/Initializer/random_uniform/subSubdrnn_classification/cell_2/bidirectional_rnn/fw/basic_lstm_cell/kernel/Initializer/random_uniform/maxdrnn_classification/cell_2/bidirectional_rnn/fw/basic_lstm_cell/kernel/Initializer/random_uniform/min*
T0*X
_classN
LJloc:@rnn_classification/cell_2/bidirectional_rnn/fw/basic_lstm_cell/kernel*
_output_shapes
: 
Ć
drnn_classification/cell_2/bidirectional_rnn/fw/basic_lstm_cell/kernel/Initializer/random_uniform/mulMulnrnn_classification/cell_2/bidirectional_rnn/fw/basic_lstm_cell/kernel/Initializer/random_uniform/RandomUniformdrnn_classification/cell_2/bidirectional_rnn/fw/basic_lstm_cell/kernel/Initializer/random_uniform/sub* 
_output_shapes
:
*
T0*X
_classN
LJloc:@rnn_classification/cell_2/bidirectional_rnn/fw/basic_lstm_cell/kernel
¸
`rnn_classification/cell_2/bidirectional_rnn/fw/basic_lstm_cell/kernel/Initializer/random_uniformAdddrnn_classification/cell_2/bidirectional_rnn/fw/basic_lstm_cell/kernel/Initializer/random_uniform/muldrnn_classification/cell_2/bidirectional_rnn/fw/basic_lstm_cell/kernel/Initializer/random_uniform/min* 
_output_shapes
:
*
T0*X
_classN
LJloc:@rnn_classification/cell_2/bidirectional_rnn/fw/basic_lstm_cell/kernel
ó
Ernn_classification/cell_2/bidirectional_rnn/fw/basic_lstm_cell/kernel
VariableV2*
shape:
*
dtype0* 
_output_shapes
:
*X
_classN
LJloc:@rnn_classification/cell_2/bidirectional_rnn/fw/basic_lstm_cell/kernel

Lrnn_classification/cell_2/bidirectional_rnn/fw/basic_lstm_cell/kernel/AssignAssignErnn_classification/cell_2/bidirectional_rnn/fw/basic_lstm_cell/kernel`rnn_classification/cell_2/bidirectional_rnn/fw/basic_lstm_cell/kernel/Initializer/random_uniform*
T0*X
_classN
LJloc:@rnn_classification/cell_2/bidirectional_rnn/fw/basic_lstm_cell/kernel* 
_output_shapes
:

Č
Jrnn_classification/cell_2/bidirectional_rnn/fw/basic_lstm_cell/kernel/readIdentityErnn_classification/cell_2/bidirectional_rnn/fw/basic_lstm_cell/kernel*
T0* 
_output_shapes
:

ü
Urnn_classification/cell_2/bidirectional_rnn/fw/basic_lstm_cell/bias/Initializer/zerosConst*
valueB*    *V
_classL
JHloc:@rnn_classification/cell_2/bidirectional_rnn/fw/basic_lstm_cell/bias*
dtype0*
_output_shapes	
:
ĺ
Crnn_classification/cell_2/bidirectional_rnn/fw/basic_lstm_cell/bias
VariableV2*
dtype0*
_output_shapes	
:*V
_classL
JHloc:@rnn_classification/cell_2/bidirectional_rnn/fw/basic_lstm_cell/bias*
shape:
î
Jrnn_classification/cell_2/bidirectional_rnn/fw/basic_lstm_cell/bias/AssignAssignCrnn_classification/cell_2/bidirectional_rnn/fw/basic_lstm_cell/biasUrnn_classification/cell_2/bidirectional_rnn/fw/basic_lstm_cell/bias/Initializer/zeros*
T0*V
_classL
JHloc:@rnn_classification/cell_2/bidirectional_rnn/fw/basic_lstm_cell/bias*
_output_shapes	
:
ż
Hrnn_classification/cell_2/bidirectional_rnn/fw/basic_lstm_cell/bias/readIdentityCrnn_classification/cell_2/bidirectional_rnn/fw/basic_lstm_cell/bias*
T0*
_output_shapes	
:
Ň
Mrnn_classification/cell_2/bidirectional_rnn/fw/fw/while/basic_lstm_cell/ConstConstA^rnn_classification/cell_2/bidirectional_rnn/fw/fw/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
Ř
Srnn_classification/cell_2/bidirectional_rnn/fw/fw/while/basic_lstm_cell/concat/axisConstA^rnn_classification/cell_2/bidirectional_rnn/fw/fw/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
ń
Nrnn_classification/cell_2/bidirectional_rnn/fw/fw/while/basic_lstm_cell/concatConcatV2Irnn_classification/cell_2/bidirectional_rnn/fw/fw/while/TensorArrayReadV3Brnn_classification/cell_2/bidirectional_rnn/fw/fw/while/Identity_4Srnn_classification/cell_2/bidirectional_rnn/fw/fw/while/basic_lstm_cell/concat/axis*
T0*
N*
_output_shapes
:	
¨
Nrnn_classification/cell_2/bidirectional_rnn/fw/fw/while/basic_lstm_cell/MatMulMatMulNrnn_classification/cell_2/bidirectional_rnn/fw/fw/while/basic_lstm_cell/concatTrnn_classification/cell_2/bidirectional_rnn/fw/fw/while/basic_lstm_cell/MatMul/Enter*
_output_shapes
:	*
T0
Ů
Trnn_classification/cell_2/bidirectional_rnn/fw/fw/while/basic_lstm_cell/MatMul/EnterEnterJrnn_classification/cell_2/bidirectional_rnn/fw/basic_lstm_cell/kernel/read*
parallel_iterations *U

frame_nameGErnn_classification/cell_2/bidirectional_rnn/fw/fw/while/while_context* 
_output_shapes
:
*
T0*
is_constant(
Ť
Ornn_classification/cell_2/bidirectional_rnn/fw/fw/while/basic_lstm_cell/BiasAddBiasAddNrnn_classification/cell_2/bidirectional_rnn/fw/fw/while/basic_lstm_cell/MatMulUrnn_classification/cell_2/bidirectional_rnn/fw/fw/while/basic_lstm_cell/BiasAdd/Enter*
_output_shapes
:	*
T0
Ó
Urnn_classification/cell_2/bidirectional_rnn/fw/fw/while/basic_lstm_cell/BiasAdd/EnterEnterHrnn_classification/cell_2/bidirectional_rnn/fw/basic_lstm_cell/bias/read*
T0*
is_constant(*
parallel_iterations *U

frame_nameGErnn_classification/cell_2/bidirectional_rnn/fw/fw/while/while_context*
_output_shapes	
:
Ô
Ornn_classification/cell_2/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Const_1ConstA^rnn_classification/cell_2/bidirectional_rnn/fw/fw/while/Identity*
dtype0*
_output_shapes
: *
value	B :
Ň
Mrnn_classification/cell_2/bidirectional_rnn/fw/fw/while/basic_lstm_cell/splitSplitMrnn_classification/cell_2/bidirectional_rnn/fw/fw/while/basic_lstm_cell/ConstOrnn_classification/cell_2/bidirectional_rnn/fw/fw/while/basic_lstm_cell/BiasAdd*
	num_split*@
_output_shapes.
,:	:	:	:	*
T0
×
Ornn_classification/cell_2/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Const_2ConstA^rnn_classification/cell_2/bidirectional_rnn/fw/fw/while/Identity*
valueB
 *  ?*
dtype0*
_output_shapes
: 

Krnn_classification/cell_2/bidirectional_rnn/fw/fw/while/basic_lstm_cell/AddAddOrnn_classification/cell_2/bidirectional_rnn/fw/fw/while/basic_lstm_cell/split:2Ornn_classification/cell_2/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Const_2*
T0*
_output_shapes
:	
Ń
Ornn_classification/cell_2/bidirectional_rnn/fw/fw/while/basic_lstm_cell/SigmoidSigmoidKrnn_classification/cell_2/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Add*
_output_shapes
:	*
T0

Krnn_classification/cell_2/bidirectional_rnn/fw/fw/while/basic_lstm_cell/MulMulBrnn_classification/cell_2/bidirectional_rnn/fw/fw/while/Identity_3Ornn_classification/cell_2/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Sigmoid*
T0*
_output_shapes
:	
Ő
Qrnn_classification/cell_2/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Sigmoid_1SigmoidMrnn_classification/cell_2/bidirectional_rnn/fw/fw/while/basic_lstm_cell/split*
T0*
_output_shapes
:	
Ď
Lrnn_classification/cell_2/bidirectional_rnn/fw/fw/while/basic_lstm_cell/TanhTanhOrnn_classification/cell_2/bidirectional_rnn/fw/fw/while/basic_lstm_cell/split:1*
_output_shapes
:	*
T0

Mrnn_classification/cell_2/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_1MulQrnn_classification/cell_2/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Sigmoid_1Lrnn_classification/cell_2/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Tanh*
T0*
_output_shapes
:	

Mrnn_classification/cell_2/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Add_1AddKrnn_classification/cell_2/bidirectional_rnn/fw/fw/while/basic_lstm_cell/MulMrnn_classification/cell_2/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_1*
T0*
_output_shapes
:	
Ď
Nrnn_classification/cell_2/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Tanh_1TanhMrnn_classification/cell_2/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Add_1*
T0*
_output_shapes
:	
×
Qrnn_classification/cell_2/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Sigmoid_2SigmoidOrnn_classification/cell_2/bidirectional_rnn/fw/fw/while/basic_lstm_cell/split:3*
T0*
_output_shapes
:	
Ą
Mrnn_classification/cell_2/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_2MulNrnn_classification/cell_2/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Tanh_1Qrnn_classification/cell_2/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Sigmoid_2*
_output_shapes
:	*
T0
Ż
>rnn_classification/cell_2/bidirectional_rnn/fw/fw/while/SelectSelectDrnn_classification/cell_2/bidirectional_rnn/fw/fw/while/GreaterEqualDrnn_classification/cell_2/bidirectional_rnn/fw/fw/while/Select/EnterMrnn_classification/cell_2/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_2*
T0*`
_classV
TRloc:@rnn_classification/cell_2/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_2*
_output_shapes
:	

Drnn_classification/cell_2/bidirectional_rnn/fw/fw/while/Select/EnterEnter7rnn_classification/cell_2/bidirectional_rnn/fw/fw/zeros*
parallel_iterations *
is_constant(*
_output_shapes
:	*U

frame_nameGErnn_classification/cell_2/bidirectional_rnn/fw/fw/while/while_context*
T0*`
_classV
TRloc:@rnn_classification/cell_2/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_2
Ż
@rnn_classification/cell_2/bidirectional_rnn/fw/fw/while/Select_1SelectDrnn_classification/cell_2/bidirectional_rnn/fw/fw/while/GreaterEqualBrnn_classification/cell_2/bidirectional_rnn/fw/fw/while/Identity_3Mrnn_classification/cell_2/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Add_1*
T0*`
_classV
TRloc:@rnn_classification/cell_2/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Add_1*
_output_shapes
:	
Ż
@rnn_classification/cell_2/bidirectional_rnn/fw/fw/while/Select_2SelectDrnn_classification/cell_2/bidirectional_rnn/fw/fw/while/GreaterEqualBrnn_classification/cell_2/bidirectional_rnn/fw/fw/while/Identity_4Mrnn_classification/cell_2/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_2*
T0*`
_classV
TRloc:@rnn_classification/cell_2/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_2*
_output_shapes
:	

[rnn_classification/cell_2/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3TensorArrayWriteV3arnn_classification/cell_2/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3/EnterBrnn_classification/cell_2/bidirectional_rnn/fw/fw/while/Identity_1>rnn_classification/cell_2/bidirectional_rnn/fw/fw/while/SelectBrnn_classification/cell_2/bidirectional_rnn/fw/fw/while/Identity_2*
T0*`
_classV
TRloc:@rnn_classification/cell_2/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_2*
_output_shapes
: 
ľ
arnn_classification/cell_2/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3/EnterEnter=rnn_classification/cell_2/bidirectional_rnn/fw/fw/TensorArray*
is_constant(*
_output_shapes
:*U

frame_nameGErnn_classification/cell_2/bidirectional_rnn/fw/fw/while/while_context*
T0*`
_classV
TRloc:@rnn_classification/cell_2/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_2*
parallel_iterations 
Ä
?rnn_classification/cell_2/bidirectional_rnn/fw/fw/while/add_1/yConstA^rnn_classification/cell_2/bidirectional_rnn/fw/fw/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
ę
=rnn_classification/cell_2/bidirectional_rnn/fw/fw/while/add_1AddBrnn_classification/cell_2/bidirectional_rnn/fw/fw/while/Identity_1?rnn_classification/cell_2/bidirectional_rnn/fw/fw/while/add_1/y*
_output_shapes
: *
T0
´
Ernn_classification/cell_2/bidirectional_rnn/fw/fw/while/NextIterationNextIteration;rnn_classification/cell_2/bidirectional_rnn/fw/fw/while/add*
_output_shapes
: *
T0
¸
Grnn_classification/cell_2/bidirectional_rnn/fw/fw/while/NextIteration_1NextIteration=rnn_classification/cell_2/bidirectional_rnn/fw/fw/while/add_1*
T0*
_output_shapes
: 
Ö
Grnn_classification/cell_2/bidirectional_rnn/fw/fw/while/NextIteration_2NextIteration[rnn_classification/cell_2/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3*
T0*
_output_shapes
: 
Ä
Grnn_classification/cell_2/bidirectional_rnn/fw/fw/while/NextIteration_3NextIteration@rnn_classification/cell_2/bidirectional_rnn/fw/fw/while/Select_1*
T0*
_output_shapes
:	
Ä
Grnn_classification/cell_2/bidirectional_rnn/fw/fw/while/NextIteration_4NextIteration@rnn_classification/cell_2/bidirectional_rnn/fw/fw/while/Select_2*
T0*
_output_shapes
:	
Ľ
<rnn_classification/cell_2/bidirectional_rnn/fw/fw/while/ExitExit>rnn_classification/cell_2/bidirectional_rnn/fw/fw/while/Switch*
T0*
_output_shapes
: 
Š
>rnn_classification/cell_2/bidirectional_rnn/fw/fw/while/Exit_1Exit@rnn_classification/cell_2/bidirectional_rnn/fw/fw/while/Switch_1*
T0*
_output_shapes
: 
Š
>rnn_classification/cell_2/bidirectional_rnn/fw/fw/while/Exit_2Exit@rnn_classification/cell_2/bidirectional_rnn/fw/fw/while/Switch_2*
T0*
_output_shapes
: 
˛
>rnn_classification/cell_2/bidirectional_rnn/fw/fw/while/Exit_3Exit@rnn_classification/cell_2/bidirectional_rnn/fw/fw/while/Switch_3*
T0*
_output_shapes
:	
˛
>rnn_classification/cell_2/bidirectional_rnn/fw/fw/while/Exit_4Exit@rnn_classification/cell_2/bidirectional_rnn/fw/fw/while/Switch_4*
T0*
_output_shapes
:	
Ň
Trnn_classification/cell_2/bidirectional_rnn/fw/fw/TensorArrayStack/TensorArraySizeV3TensorArraySizeV3=rnn_classification/cell_2/bidirectional_rnn/fw/fw/TensorArray>rnn_classification/cell_2/bidirectional_rnn/fw/fw/while/Exit_2*P
_classF
DBloc:@rnn_classification/cell_2/bidirectional_rnn/fw/fw/TensorArray*
_output_shapes
: 
â
Nrnn_classification/cell_2/bidirectional_rnn/fw/fw/TensorArrayStack/range/startConst*
value	B : *P
_classF
DBloc:@rnn_classification/cell_2/bidirectional_rnn/fw/fw/TensorArray*
dtype0*
_output_shapes
: 
â
Nrnn_classification/cell_2/bidirectional_rnn/fw/fw/TensorArrayStack/range/deltaConst*
dtype0*
_output_shapes
: *
value	B :*P
_classF
DBloc:@rnn_classification/cell_2/bidirectional_rnn/fw/fw/TensorArray
ž
Hrnn_classification/cell_2/bidirectional_rnn/fw/fw/TensorArrayStack/rangeRangeNrnn_classification/cell_2/bidirectional_rnn/fw/fw/TensorArrayStack/range/startTrnn_classification/cell_2/bidirectional_rnn/fw/fw/TensorArrayStack/TensorArraySizeV3Nrnn_classification/cell_2/bidirectional_rnn/fw/fw/TensorArrayStack/range/delta*P
_classF
DBloc:@rnn_classification/cell_2/bidirectional_rnn/fw/fw/TensorArray*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
á
Vrnn_classification/cell_2/bidirectional_rnn/fw/fw/TensorArrayStack/TensorArrayGatherV3TensorArrayGatherV3=rnn_classification/cell_2/bidirectional_rnn/fw/fw/TensorArrayHrnn_classification/cell_2/bidirectional_rnn/fw/fw/TensorArrayStack/range>rnn_classification/cell_2/bidirectional_rnn/fw/fw/while/Exit_2*P
_classF
DBloc:@rnn_classification/cell_2/bidirectional_rnn/fw/fw/TensorArray*
dtype0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙*
element_shape:	

9rnn_classification/cell_2/bidirectional_rnn/fw/fw/Const_5Const*
valueB:*
dtype0*
_output_shapes
:
z
8rnn_classification/cell_2/bidirectional_rnn/fw/fw/Rank_1Const*
dtype0*
_output_shapes
: *
value	B :

?rnn_classification/cell_2/bidirectional_rnn/fw/fw/range_1/startConst*
value	B :*
dtype0*
_output_shapes
: 

?rnn_classification/cell_2/bidirectional_rnn/fw/fw/range_1/deltaConst*
value	B :*
dtype0*
_output_shapes
: 

9rnn_classification/cell_2/bidirectional_rnn/fw/fw/range_1Range?rnn_classification/cell_2/bidirectional_rnn/fw/fw/range_1/start8rnn_classification/cell_2/bidirectional_rnn/fw/fw/Rank_1?rnn_classification/cell_2/bidirectional_rnn/fw/fw/range_1/delta*
_output_shapes
:

Crnn_classification/cell_2/bidirectional_rnn/fw/fw/concat_2/values_0Const*
valueB"       *
dtype0*
_output_shapes
:

?rnn_classification/cell_2/bidirectional_rnn/fw/fw/concat_2/axisConst*
value	B : *
dtype0*
_output_shapes
: 
ľ
:rnn_classification/cell_2/bidirectional_rnn/fw/fw/concat_2ConcatV2Crnn_classification/cell_2/bidirectional_rnn/fw/fw/concat_2/values_09rnn_classification/cell_2/bidirectional_rnn/fw/fw/range_1?rnn_classification/cell_2/bidirectional_rnn/fw/fw/concat_2/axis*
T0*
N*
_output_shapes
:

=rnn_classification/cell_2/bidirectional_rnn/fw/fw/transpose_1	TransposeVrnn_classification/cell_2/bidirectional_rnn/fw/fw/TensorArrayStack/TensorArrayGatherV3:rnn_classification/cell_2/bidirectional_rnn/fw/fw/concat_2*,
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Â
>rnn_classification/cell_2/bidirectional_rnn/bw/ReverseSequenceReverseSequence rnn_classification/cell_1/concatSqueeze*
seq_dim*,
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
x
6rnn_classification/cell_2/bidirectional_rnn/bw/bw/RankConst*
value	B :*
dtype0*
_output_shapes
: 

=rnn_classification/cell_2/bidirectional_rnn/bw/bw/range/startConst*
value	B :*
dtype0*
_output_shapes
: 

=rnn_classification/cell_2/bidirectional_rnn/bw/bw/range/deltaConst*
dtype0*
_output_shapes
: *
value	B :

7rnn_classification/cell_2/bidirectional_rnn/bw/bw/rangeRange=rnn_classification/cell_2/bidirectional_rnn/bw/bw/range/start6rnn_classification/cell_2/bidirectional_rnn/bw/bw/Rank=rnn_classification/cell_2/bidirectional_rnn/bw/bw/range/delta*
_output_shapes
:

Arnn_classification/cell_2/bidirectional_rnn/bw/bw/concat/values_0Const*
valueB"       *
dtype0*
_output_shapes
:

=rnn_classification/cell_2/bidirectional_rnn/bw/bw/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
­
8rnn_classification/cell_2/bidirectional_rnn/bw/bw/concatConcatV2Arnn_classification/cell_2/bidirectional_rnn/bw/bw/concat/values_07rnn_classification/cell_2/bidirectional_rnn/bw/bw/range=rnn_classification/cell_2/bidirectional_rnn/bw/bw/concat/axis*
T0*
N*
_output_shapes
:
ů
;rnn_classification/cell_2/bidirectional_rnn/bw/bw/transpose	Transpose>rnn_classification/cell_2/bidirectional_rnn/bw/ReverseSequence8rnn_classification/cell_2/bidirectional_rnn/bw/bw/concat*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙
~
9rnn_classification/cell_2/bidirectional_rnn/bw/bw/ToInt32CastSqueeze*

SrcT0	*

DstT0*
_output_shapes
:
­
Arnn_classification/cell_2/bidirectional_rnn/bw/bw/sequence_lengthIdentity9rnn_classification/cell_2/bidirectional_rnn/bw/bw/ToInt32*
T0*
_output_shapes
:
°
frnn_classification/cell_2/bidirectional_rnn/bw/bw/DropoutWrapperZeroState/BasicLSTMCellZeroState/ConstConst*
valueB:*
dtype0*
_output_shapes
:
ł
hrnn_classification/cell_2/bidirectional_rnn/bw/bw/DropoutWrapperZeroState/BasicLSTMCellZeroState/Const_1Const*
valueB:*
dtype0*
_output_shapes
:
Ž
lrnn_classification/cell_2/bidirectional_rnn/bw/bw/DropoutWrapperZeroState/BasicLSTMCellZeroState/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
á
grnn_classification/cell_2/bidirectional_rnn/bw/bw/DropoutWrapperZeroState/BasicLSTMCellZeroState/concatConcatV2frnn_classification/cell_2/bidirectional_rnn/bw/bw/DropoutWrapperZeroState/BasicLSTMCellZeroState/Consthrnn_classification/cell_2/bidirectional_rnn/bw/bw/DropoutWrapperZeroState/BasicLSTMCellZeroState/Const_1lrnn_classification/cell_2/bidirectional_rnn/bw/bw/DropoutWrapperZeroState/BasicLSTMCellZeroState/concat/axis*
N*
_output_shapes
:*
T0
ą
lrnn_classification/cell_2/bidirectional_rnn/bw/bw/DropoutWrapperZeroState/BasicLSTMCellZeroState/zeros/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    
ď
frnn_classification/cell_2/bidirectional_rnn/bw/bw/DropoutWrapperZeroState/BasicLSTMCellZeroState/zerosFillgrnn_classification/cell_2/bidirectional_rnn/bw/bw/DropoutWrapperZeroState/BasicLSTMCellZeroState/concatlrnn_classification/cell_2/bidirectional_rnn/bw/bw/DropoutWrapperZeroState/BasicLSTMCellZeroState/zeros/Const*
T0*
_output_shapes
:	
˛
hrnn_classification/cell_2/bidirectional_rnn/bw/bw/DropoutWrapperZeroState/BasicLSTMCellZeroState/Const_2Const*
dtype0*
_output_shapes
:*
valueB:
ł
hrnn_classification/cell_2/bidirectional_rnn/bw/bw/DropoutWrapperZeroState/BasicLSTMCellZeroState/Const_3Const*
valueB:*
dtype0*
_output_shapes
:
˛
hrnn_classification/cell_2/bidirectional_rnn/bw/bw/DropoutWrapperZeroState/BasicLSTMCellZeroState/Const_4Const*
valueB:*
dtype0*
_output_shapes
:
ł
hrnn_classification/cell_2/bidirectional_rnn/bw/bw/DropoutWrapperZeroState/BasicLSTMCellZeroState/Const_5Const*
valueB:*
dtype0*
_output_shapes
:
°
nrnn_classification/cell_2/bidirectional_rnn/bw/bw/DropoutWrapperZeroState/BasicLSTMCellZeroState/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
ç
irnn_classification/cell_2/bidirectional_rnn/bw/bw/DropoutWrapperZeroState/BasicLSTMCellZeroState/concat_1ConcatV2hrnn_classification/cell_2/bidirectional_rnn/bw/bw/DropoutWrapperZeroState/BasicLSTMCellZeroState/Const_4hrnn_classification/cell_2/bidirectional_rnn/bw/bw/DropoutWrapperZeroState/BasicLSTMCellZeroState/Const_5nrnn_classification/cell_2/bidirectional_rnn/bw/bw/DropoutWrapperZeroState/BasicLSTMCellZeroState/concat_1/axis*
T0*
N*
_output_shapes
:
ł
nrnn_classification/cell_2/bidirectional_rnn/bw/bw/DropoutWrapperZeroState/BasicLSTMCellZeroState/zeros_1/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
ő
hrnn_classification/cell_2/bidirectional_rnn/bw/bw/DropoutWrapperZeroState/BasicLSTMCellZeroState/zeros_1Fillirnn_classification/cell_2/bidirectional_rnn/bw/bw/DropoutWrapperZeroState/BasicLSTMCellZeroState/concat_1nrnn_classification/cell_2/bidirectional_rnn/bw/bw/DropoutWrapperZeroState/BasicLSTMCellZeroState/zeros_1/Const*
T0*
_output_shapes
:	
˛
hrnn_classification/cell_2/bidirectional_rnn/bw/bw/DropoutWrapperZeroState/BasicLSTMCellZeroState/Const_6Const*
valueB:*
dtype0*
_output_shapes
:
ł
hrnn_classification/cell_2/bidirectional_rnn/bw/bw/DropoutWrapperZeroState/BasicLSTMCellZeroState/Const_7Const*
valueB:*
dtype0*
_output_shapes
:

7rnn_classification/cell_2/bidirectional_rnn/bw/bw/ShapeConst*
valueB:*
dtype0*
_output_shapes
:

7rnn_classification/cell_2/bidirectional_rnn/bw/bw/stackConst*
valueB:*
dtype0*
_output_shapes
:
×
7rnn_classification/cell_2/bidirectional_rnn/bw/bw/EqualEqual7rnn_classification/cell_2/bidirectional_rnn/bw/bw/Shape7rnn_classification/cell_2/bidirectional_rnn/bw/bw/stack*
T0*
_output_shapes
:

7rnn_classification/cell_2/bidirectional_rnn/bw/bw/ConstConst*
dtype0*
_output_shapes
:*
valueB: 
Ć
5rnn_classification/cell_2/bidirectional_rnn/bw/bw/AllAll7rnn_classification/cell_2/bidirectional_rnn/bw/bw/Equal7rnn_classification/cell_2/bidirectional_rnn/bw/bw/Const*
_output_shapes
: 
ŕ
>rnn_classification/cell_2/bidirectional_rnn/bw/bw/Assert/ConstConst*r
valueiBg BaExpected shape for Tensor rnn_classification/cell_2/bidirectional_rnn/bw/bw/sequence_length:0 is *
dtype0*
_output_shapes
: 

@rnn_classification/cell_2/bidirectional_rnn/bw/bw/Assert/Const_1Const*
dtype0*
_output_shapes
: *!
valueB B but saw shape: 
č
Frnn_classification/cell_2/bidirectional_rnn/bw/bw/Assert/Assert/data_0Const*r
valueiBg BaExpected shape for Tensor rnn_classification/cell_2/bidirectional_rnn/bw/bw/sequence_length:0 is *
dtype0*
_output_shapes
: 

Frnn_classification/cell_2/bidirectional_rnn/bw/bw/Assert/Assert/data_2Const*!
valueB B but saw shape: *
dtype0*
_output_shapes
: 

?rnn_classification/cell_2/bidirectional_rnn/bw/bw/Assert/AssertAssert5rnn_classification/cell_2/bidirectional_rnn/bw/bw/AllFrnn_classification/cell_2/bidirectional_rnn/bw/bw/Assert/Assert/data_07rnn_classification/cell_2/bidirectional_rnn/bw/bw/stackFrnn_classification/cell_2/bidirectional_rnn/bw/bw/Assert/Assert/data_27rnn_classification/cell_2/bidirectional_rnn/bw/bw/Shape*
T
2
ó
=rnn_classification/cell_2/bidirectional_rnn/bw/bw/CheckSeqLenIdentityArnn_classification/cell_2/bidirectional_rnn/bw/bw/sequence_length@^rnn_classification/cell_2/bidirectional_rnn/bw/bw/Assert/Assert*
T0*
_output_shapes
:
¤
9rnn_classification/cell_2/bidirectional_rnn/bw/bw/Shape_1Shape;rnn_classification/cell_2/bidirectional_rnn/bw/bw/transpose*
_output_shapes
:*
T0

Ernn_classification/cell_2/bidirectional_rnn/bw/bw/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:

Grnn_classification/cell_2/bidirectional_rnn/bw/bw/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:

Grnn_classification/cell_2/bidirectional_rnn/bw/bw/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
Š
?rnn_classification/cell_2/bidirectional_rnn/bw/bw/strided_sliceStridedSlice9rnn_classification/cell_2/bidirectional_rnn/bw/bw/Shape_1Ernn_classification/cell_2/bidirectional_rnn/bw/bw/strided_slice/stackGrnn_classification/cell_2/bidirectional_rnn/bw/bw/strided_slice/stack_1Grnn_classification/cell_2/bidirectional_rnn/bw/bw/strided_slice/stack_2*
_output_shapes
: *
T0*
Index0*
shrink_axis_mask

9rnn_classification/cell_2/bidirectional_rnn/bw/bw/Const_1Const*
valueB:*
dtype0*
_output_shapes
:

9rnn_classification/cell_2/bidirectional_rnn/bw/bw/Const_2Const*
dtype0*
_output_shapes
:*
valueB:

?rnn_classification/cell_2/bidirectional_rnn/bw/bw/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
Ť
:rnn_classification/cell_2/bidirectional_rnn/bw/bw/concat_1ConcatV29rnn_classification/cell_2/bidirectional_rnn/bw/bw/Const_19rnn_classification/cell_2/bidirectional_rnn/bw/bw/Const_2?rnn_classification/cell_2/bidirectional_rnn/bw/bw/concat_1/axis*
N*
_output_shapes
:*
T0

=rnn_classification/cell_2/bidirectional_rnn/bw/bw/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
ä
7rnn_classification/cell_2/bidirectional_rnn/bw/bw/zerosFill:rnn_classification/cell_2/bidirectional_rnn/bw/bw/concat_1=rnn_classification/cell_2/bidirectional_rnn/bw/bw/zeros/Const*
_output_shapes
:	*
T0

9rnn_classification/cell_2/bidirectional_rnn/bw/bw/Const_3Const*
valueB: *
dtype0*
_output_shapes
:
×
5rnn_classification/cell_2/bidirectional_rnn/bw/bw/MinMin=rnn_classification/cell_2/bidirectional_rnn/bw/bw/CheckSeqLen9rnn_classification/cell_2/bidirectional_rnn/bw/bw/Const_3*
T0*
_output_shapes
: 

9rnn_classification/cell_2/bidirectional_rnn/bw/bw/Const_4Const*
valueB: *
dtype0*
_output_shapes
:
×
5rnn_classification/cell_2/bidirectional_rnn/bw/bw/MaxMax=rnn_classification/cell_2/bidirectional_rnn/bw/bw/CheckSeqLen9rnn_classification/cell_2/bidirectional_rnn/bw/bw/Const_4*
T0*
_output_shapes
: 
x
6rnn_classification/cell_2/bidirectional_rnn/bw/bw/timeConst*
dtype0*
_output_shapes
: *
value	B : 
×
=rnn_classification/cell_2/bidirectional_rnn/bw/bw/TensorArrayTensorArrayV3?rnn_classification/cell_2/bidirectional_rnn/bw/bw/strided_slice*]
tensor_array_nameHFrnn_classification/cell_2/bidirectional_rnn/bw/bw/dynamic_rnn/output_0*
dtype0*
_output_shapes

:: *
element_shape:	*
identical_element_shapes(
Ř
?rnn_classification/cell_2/bidirectional_rnn/bw/bw/TensorArray_1TensorArrayV3?rnn_classification/cell_2/bidirectional_rnn/bw/bw/strided_slice*\
tensor_array_nameGErnn_classification/cell_2/bidirectional_rnn/bw/bw/dynamic_rnn/input_0*
dtype0*
_output_shapes

:: *
element_shape:	*
identical_element_shapes(
ľ
Jrnn_classification/cell_2/bidirectional_rnn/bw/bw/TensorArrayUnstack/ShapeShape;rnn_classification/cell_2/bidirectional_rnn/bw/bw/transpose*
T0*
_output_shapes
:
˘
Xrnn_classification/cell_2/bidirectional_rnn/bw/bw/TensorArrayUnstack/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
¤
Zrnn_classification/cell_2/bidirectional_rnn/bw/bw/TensorArrayUnstack/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
¤
Zrnn_classification/cell_2/bidirectional_rnn/bw/bw/TensorArrayUnstack/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:

Rrnn_classification/cell_2/bidirectional_rnn/bw/bw/TensorArrayUnstack/strided_sliceStridedSliceJrnn_classification/cell_2/bidirectional_rnn/bw/bw/TensorArrayUnstack/ShapeXrnn_classification/cell_2/bidirectional_rnn/bw/bw/TensorArrayUnstack/strided_slice/stackZrnn_classification/cell_2/bidirectional_rnn/bw/bw/TensorArrayUnstack/strided_slice/stack_1Zrnn_classification/cell_2/bidirectional_rnn/bw/bw/TensorArrayUnstack/strided_slice/stack_2*
_output_shapes
: *
Index0*
T0*
shrink_axis_mask

Prnn_classification/cell_2/bidirectional_rnn/bw/bw/TensorArrayUnstack/range/startConst*
value	B : *
dtype0*
_output_shapes
: 

Prnn_classification/cell_2/bidirectional_rnn/bw/bw/TensorArrayUnstack/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
đ
Jrnn_classification/cell_2/bidirectional_rnn/bw/bw/TensorArrayUnstack/rangeRangePrnn_classification/cell_2/bidirectional_rnn/bw/bw/TensorArrayUnstack/range/startRrnn_classification/cell_2/bidirectional_rnn/bw/bw/TensorArrayUnstack/strided_slicePrnn_classification/cell_2/bidirectional_rnn/bw/bw/TensorArrayUnstack/range/delta*#
_output_shapes
:˙˙˙˙˙˙˙˙˙

lrnn_classification/cell_2/bidirectional_rnn/bw/bw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3TensorArrayScatterV3?rnn_classification/cell_2/bidirectional_rnn/bw/bw/TensorArray_1Jrnn_classification/cell_2/bidirectional_rnn/bw/bw/TensorArrayUnstack/range;rnn_classification/cell_2/bidirectional_rnn/bw/bw/transposeArnn_classification/cell_2/bidirectional_rnn/bw/bw/TensorArray_1:1*
T0*N
_classD
B@loc:@rnn_classification/cell_2/bidirectional_rnn/bw/bw/transpose*
_output_shapes
: 
}
;rnn_classification/cell_2/bidirectional_rnn/bw/bw/Maximum/xConst*
value	B :*
dtype0*
_output_shapes
: 
Ů
9rnn_classification/cell_2/bidirectional_rnn/bw/bw/MaximumMaximum;rnn_classification/cell_2/bidirectional_rnn/bw/bw/Maximum/x5rnn_classification/cell_2/bidirectional_rnn/bw/bw/Max*
T0*
_output_shapes
: 
á
9rnn_classification/cell_2/bidirectional_rnn/bw/bw/MinimumMinimum?rnn_classification/cell_2/bidirectional_rnn/bw/bw/strided_slice9rnn_classification/cell_2/bidirectional_rnn/bw/bw/Maximum*
T0*
_output_shapes
: 

Irnn_classification/cell_2/bidirectional_rnn/bw/bw/while/iteration_counterConst*
value	B : *
dtype0*
_output_shapes
: 
¤
=rnn_classification/cell_2/bidirectional_rnn/bw/bw/while/EnterEnterIrnn_classification/cell_2/bidirectional_rnn/bw/bw/while/iteration_counter*
T0*
parallel_iterations *U

frame_nameGErnn_classification/cell_2/bidirectional_rnn/bw/bw/while/while_context*
_output_shapes
: 

?rnn_classification/cell_2/bidirectional_rnn/bw/bw/while/Enter_1Enter6rnn_classification/cell_2/bidirectional_rnn/bw/bw/time*
parallel_iterations *U

frame_nameGErnn_classification/cell_2/bidirectional_rnn/bw/bw/while/while_context*
_output_shapes
: *
T0

?rnn_classification/cell_2/bidirectional_rnn/bw/bw/while/Enter_2Enter?rnn_classification/cell_2/bidirectional_rnn/bw/bw/TensorArray:1*
T0*
parallel_iterations *U

frame_nameGErnn_classification/cell_2/bidirectional_rnn/bw/bw/while/while_context*
_output_shapes
: 
Ě
?rnn_classification/cell_2/bidirectional_rnn/bw/bw/while/Enter_3Enterfrnn_classification/cell_2/bidirectional_rnn/bw/bw/DropoutWrapperZeroState/BasicLSTMCellZeroState/zeros*
T0*
parallel_iterations *U

frame_nameGErnn_classification/cell_2/bidirectional_rnn/bw/bw/while/while_context*
_output_shapes
:	
Î
?rnn_classification/cell_2/bidirectional_rnn/bw/bw/while/Enter_4Enterhrnn_classification/cell_2/bidirectional_rnn/bw/bw/DropoutWrapperZeroState/BasicLSTMCellZeroState/zeros_1*
T0*
parallel_iterations *U

frame_nameGErnn_classification/cell_2/bidirectional_rnn/bw/bw/while/while_context*
_output_shapes
:	
ř
=rnn_classification/cell_2/bidirectional_rnn/bw/bw/while/MergeMerge=rnn_classification/cell_2/bidirectional_rnn/bw/bw/while/EnterErnn_classification/cell_2/bidirectional_rnn/bw/bw/while/NextIteration*
T0*
N*
_output_shapes
: : 
ţ
?rnn_classification/cell_2/bidirectional_rnn/bw/bw/while/Merge_1Merge?rnn_classification/cell_2/bidirectional_rnn/bw/bw/while/Enter_1Grnn_classification/cell_2/bidirectional_rnn/bw/bw/while/NextIteration_1*
T0*
N*
_output_shapes
: : 
ţ
?rnn_classification/cell_2/bidirectional_rnn/bw/bw/while/Merge_2Merge?rnn_classification/cell_2/bidirectional_rnn/bw/bw/while/Enter_2Grnn_classification/cell_2/bidirectional_rnn/bw/bw/while/NextIteration_2*
T0*
N*
_output_shapes
: : 

?rnn_classification/cell_2/bidirectional_rnn/bw/bw/while/Merge_3Merge?rnn_classification/cell_2/bidirectional_rnn/bw/bw/while/Enter_3Grnn_classification/cell_2/bidirectional_rnn/bw/bw/while/NextIteration_3*
N*!
_output_shapes
:	: *
T0

?rnn_classification/cell_2/bidirectional_rnn/bw/bw/while/Merge_4Merge?rnn_classification/cell_2/bidirectional_rnn/bw/bw/while/Enter_4Grnn_classification/cell_2/bidirectional_rnn/bw/bw/while/NextIteration_4*
N*!
_output_shapes
:	: *
T0
č
<rnn_classification/cell_2/bidirectional_rnn/bw/bw/while/LessLess=rnn_classification/cell_2/bidirectional_rnn/bw/bw/while/MergeBrnn_classification/cell_2/bidirectional_rnn/bw/bw/while/Less/Enter*
T0*
_output_shapes
: 
˛
Brnn_classification/cell_2/bidirectional_rnn/bw/bw/while/Less/EnterEnter?rnn_classification/cell_2/bidirectional_rnn/bw/bw/strided_slice*
T0*
is_constant(*
parallel_iterations *U

frame_nameGErnn_classification/cell_2/bidirectional_rnn/bw/bw/while/while_context*
_output_shapes
: 
î
>rnn_classification/cell_2/bidirectional_rnn/bw/bw/while/Less_1Less?rnn_classification/cell_2/bidirectional_rnn/bw/bw/while/Merge_1Drnn_classification/cell_2/bidirectional_rnn/bw/bw/while/Less_1/Enter*
T0*
_output_shapes
: 
Ž
Drnn_classification/cell_2/bidirectional_rnn/bw/bw/while/Less_1/EnterEnter9rnn_classification/cell_2/bidirectional_rnn/bw/bw/Minimum*
T0*
is_constant(*
parallel_iterations *U

frame_nameGErnn_classification/cell_2/bidirectional_rnn/bw/bw/while/while_context*
_output_shapes
: 
ć
Brnn_classification/cell_2/bidirectional_rnn/bw/bw/while/LogicalAnd
LogicalAnd<rnn_classification/cell_2/bidirectional_rnn/bw/bw/while/Less>rnn_classification/cell_2/bidirectional_rnn/bw/bw/while/Less_1*
_output_shapes
: 
¨
@rnn_classification/cell_2/bidirectional_rnn/bw/bw/while/LoopCondLoopCondBrnn_classification/cell_2/bidirectional_rnn/bw/bw/while/LogicalAnd*
_output_shapes
: 
ž
>rnn_classification/cell_2/bidirectional_rnn/bw/bw/while/SwitchSwitch=rnn_classification/cell_2/bidirectional_rnn/bw/bw/while/Merge@rnn_classification/cell_2/bidirectional_rnn/bw/bw/while/LoopCond*
_output_shapes
: : *
T0*P
_classF
DBloc:@rnn_classification/cell_2/bidirectional_rnn/bw/bw/while/Merge
Ä
@rnn_classification/cell_2/bidirectional_rnn/bw/bw/while/Switch_1Switch?rnn_classification/cell_2/bidirectional_rnn/bw/bw/while/Merge_1@rnn_classification/cell_2/bidirectional_rnn/bw/bw/while/LoopCond*
_output_shapes
: : *
T0*R
_classH
FDloc:@rnn_classification/cell_2/bidirectional_rnn/bw/bw/while/Merge_1
Ä
@rnn_classification/cell_2/bidirectional_rnn/bw/bw/while/Switch_2Switch?rnn_classification/cell_2/bidirectional_rnn/bw/bw/while/Merge_2@rnn_classification/cell_2/bidirectional_rnn/bw/bw/while/LoopCond*
T0*R
_classH
FDloc:@rnn_classification/cell_2/bidirectional_rnn/bw/bw/while/Merge_2*
_output_shapes
: : 
Ö
@rnn_classification/cell_2/bidirectional_rnn/bw/bw/while/Switch_3Switch?rnn_classification/cell_2/bidirectional_rnn/bw/bw/while/Merge_3@rnn_classification/cell_2/bidirectional_rnn/bw/bw/while/LoopCond*
T0*R
_classH
FDloc:@rnn_classification/cell_2/bidirectional_rnn/bw/bw/while/Merge_3**
_output_shapes
:	:	
Ö
@rnn_classification/cell_2/bidirectional_rnn/bw/bw/while/Switch_4Switch?rnn_classification/cell_2/bidirectional_rnn/bw/bw/while/Merge_4@rnn_classification/cell_2/bidirectional_rnn/bw/bw/while/LoopCond*
T0*R
_classH
FDloc:@rnn_classification/cell_2/bidirectional_rnn/bw/bw/while/Merge_4**
_output_shapes
:	:	
Ż
@rnn_classification/cell_2/bidirectional_rnn/bw/bw/while/IdentityIdentity@rnn_classification/cell_2/bidirectional_rnn/bw/bw/while/Switch:1*
T0*
_output_shapes
: 
ł
Brnn_classification/cell_2/bidirectional_rnn/bw/bw/while/Identity_1IdentityBrnn_classification/cell_2/bidirectional_rnn/bw/bw/while/Switch_1:1*
T0*
_output_shapes
: 
ł
Brnn_classification/cell_2/bidirectional_rnn/bw/bw/while/Identity_2IdentityBrnn_classification/cell_2/bidirectional_rnn/bw/bw/while/Switch_2:1*
T0*
_output_shapes
: 
ź
Brnn_classification/cell_2/bidirectional_rnn/bw/bw/while/Identity_3IdentityBrnn_classification/cell_2/bidirectional_rnn/bw/bw/while/Switch_3:1*
_output_shapes
:	*
T0
ź
Brnn_classification/cell_2/bidirectional_rnn/bw/bw/while/Identity_4IdentityBrnn_classification/cell_2/bidirectional_rnn/bw/bw/while/Switch_4:1*
T0*
_output_shapes
:	
Â
=rnn_classification/cell_2/bidirectional_rnn/bw/bw/while/add/yConstA^rnn_classification/cell_2/bidirectional_rnn/bw/bw/while/Identity*
dtype0*
_output_shapes
: *
value	B :
ä
;rnn_classification/cell_2/bidirectional_rnn/bw/bw/while/addAdd@rnn_classification/cell_2/bidirectional_rnn/bw/bw/while/Identity=rnn_classification/cell_2/bidirectional_rnn/bw/bw/while/add/y*
T0*
_output_shapes
: 
ô
Irnn_classification/cell_2/bidirectional_rnn/bw/bw/while/TensorArrayReadV3TensorArrayReadV3Ornn_classification/cell_2/bidirectional_rnn/bw/bw/while/TensorArrayReadV3/EnterBrnn_classification/cell_2/bidirectional_rnn/bw/bw/while/Identity_1Qrnn_classification/cell_2/bidirectional_rnn/bw/bw/while/TensorArrayReadV3/Enter_1*
dtype0*
_output_shapes
:	
Ă
Ornn_classification/cell_2/bidirectional_rnn/bw/bw/while/TensorArrayReadV3/EnterEnter?rnn_classification/cell_2/bidirectional_rnn/bw/bw/TensorArray_1*
parallel_iterations *U

frame_nameGErnn_classification/cell_2/bidirectional_rnn/bw/bw/while/while_context*
_output_shapes
:*
T0*
is_constant(
î
Qrnn_classification/cell_2/bidirectional_rnn/bw/bw/while/TensorArrayReadV3/Enter_1Enterlrnn_classification/cell_2/bidirectional_rnn/bw/bw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3*
parallel_iterations *U

frame_nameGErnn_classification/cell_2/bidirectional_rnn/bw/bw/while/while_context*
_output_shapes
: *
T0*
is_constant(

Drnn_classification/cell_2/bidirectional_rnn/bw/bw/while/GreaterEqualGreaterEqualBrnn_classification/cell_2/bidirectional_rnn/bw/bw/while/Identity_1Jrnn_classification/cell_2/bidirectional_rnn/bw/bw/while/GreaterEqual/Enter*
T0*
_output_shapes
:
ź
Jrnn_classification/cell_2/bidirectional_rnn/bw/bw/while/GreaterEqual/EnterEnter=rnn_classification/cell_2/bidirectional_rnn/bw/bw/CheckSeqLen*
parallel_iterations *U

frame_nameGErnn_classification/cell_2/bidirectional_rnn/bw/bw/while/while_context*
_output_shapes
:*
T0*
is_constant(

frnn_classification/cell_2/bidirectional_rnn/bw/basic_lstm_cell/kernel/Initializer/random_uniform/shapeConst*
valueB"     *X
_classN
LJloc:@rnn_classification/cell_2/bidirectional_rnn/bw/basic_lstm_cell/kernel*
dtype0*
_output_shapes
:

drnn_classification/cell_2/bidirectional_rnn/bw/basic_lstm_cell/kernel/Initializer/random_uniform/minConst*
valueB
 *b§˝*X
_classN
LJloc:@rnn_classification/cell_2/bidirectional_rnn/bw/basic_lstm_cell/kernel*
dtype0*
_output_shapes
: 

drnn_classification/cell_2/bidirectional_rnn/bw/basic_lstm_cell/kernel/Initializer/random_uniform/maxConst*
valueB
 *b§=*X
_classN
LJloc:@rnn_classification/cell_2/bidirectional_rnn/bw/basic_lstm_cell/kernel*
dtype0*
_output_shapes
: 
ů
nrnn_classification/cell_2/bidirectional_rnn/bw/basic_lstm_cell/kernel/Initializer/random_uniform/RandomUniformRandomUniformfrnn_classification/cell_2/bidirectional_rnn/bw/basic_lstm_cell/kernel/Initializer/random_uniform/shape*
dtype0* 
_output_shapes
:
*
T0*X
_classN
LJloc:@rnn_classification/cell_2/bidirectional_rnn/bw/basic_lstm_cell/kernel
˛
drnn_classification/cell_2/bidirectional_rnn/bw/basic_lstm_cell/kernel/Initializer/random_uniform/subSubdrnn_classification/cell_2/bidirectional_rnn/bw/basic_lstm_cell/kernel/Initializer/random_uniform/maxdrnn_classification/cell_2/bidirectional_rnn/bw/basic_lstm_cell/kernel/Initializer/random_uniform/min*
T0*X
_classN
LJloc:@rnn_classification/cell_2/bidirectional_rnn/bw/basic_lstm_cell/kernel*
_output_shapes
: 
Ć
drnn_classification/cell_2/bidirectional_rnn/bw/basic_lstm_cell/kernel/Initializer/random_uniform/mulMulnrnn_classification/cell_2/bidirectional_rnn/bw/basic_lstm_cell/kernel/Initializer/random_uniform/RandomUniformdrnn_classification/cell_2/bidirectional_rnn/bw/basic_lstm_cell/kernel/Initializer/random_uniform/sub*
T0*X
_classN
LJloc:@rnn_classification/cell_2/bidirectional_rnn/bw/basic_lstm_cell/kernel* 
_output_shapes
:

¸
`rnn_classification/cell_2/bidirectional_rnn/bw/basic_lstm_cell/kernel/Initializer/random_uniformAdddrnn_classification/cell_2/bidirectional_rnn/bw/basic_lstm_cell/kernel/Initializer/random_uniform/muldrnn_classification/cell_2/bidirectional_rnn/bw/basic_lstm_cell/kernel/Initializer/random_uniform/min* 
_output_shapes
:
*
T0*X
_classN
LJloc:@rnn_classification/cell_2/bidirectional_rnn/bw/basic_lstm_cell/kernel
ó
Ernn_classification/cell_2/bidirectional_rnn/bw/basic_lstm_cell/kernel
VariableV2*
dtype0* 
_output_shapes
:
*X
_classN
LJloc:@rnn_classification/cell_2/bidirectional_rnn/bw/basic_lstm_cell/kernel*
shape:


Lrnn_classification/cell_2/bidirectional_rnn/bw/basic_lstm_cell/kernel/AssignAssignErnn_classification/cell_2/bidirectional_rnn/bw/basic_lstm_cell/kernel`rnn_classification/cell_2/bidirectional_rnn/bw/basic_lstm_cell/kernel/Initializer/random_uniform*
T0*X
_classN
LJloc:@rnn_classification/cell_2/bidirectional_rnn/bw/basic_lstm_cell/kernel* 
_output_shapes
:

Č
Jrnn_classification/cell_2/bidirectional_rnn/bw/basic_lstm_cell/kernel/readIdentityErnn_classification/cell_2/bidirectional_rnn/bw/basic_lstm_cell/kernel*
T0* 
_output_shapes
:

ü
Urnn_classification/cell_2/bidirectional_rnn/bw/basic_lstm_cell/bias/Initializer/zerosConst*
dtype0*
_output_shapes	
:*
valueB*    *V
_classL
JHloc:@rnn_classification/cell_2/bidirectional_rnn/bw/basic_lstm_cell/bias
ĺ
Crnn_classification/cell_2/bidirectional_rnn/bw/basic_lstm_cell/bias
VariableV2*V
_classL
JHloc:@rnn_classification/cell_2/bidirectional_rnn/bw/basic_lstm_cell/bias*
shape:*
dtype0*
_output_shapes	
:
î
Jrnn_classification/cell_2/bidirectional_rnn/bw/basic_lstm_cell/bias/AssignAssignCrnn_classification/cell_2/bidirectional_rnn/bw/basic_lstm_cell/biasUrnn_classification/cell_2/bidirectional_rnn/bw/basic_lstm_cell/bias/Initializer/zeros*
T0*V
_classL
JHloc:@rnn_classification/cell_2/bidirectional_rnn/bw/basic_lstm_cell/bias*
_output_shapes	
:
ż
Hrnn_classification/cell_2/bidirectional_rnn/bw/basic_lstm_cell/bias/readIdentityCrnn_classification/cell_2/bidirectional_rnn/bw/basic_lstm_cell/bias*
_output_shapes	
:*
T0
Ň
Mrnn_classification/cell_2/bidirectional_rnn/bw/bw/while/basic_lstm_cell/ConstConstA^rnn_classification/cell_2/bidirectional_rnn/bw/bw/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
Ř
Srnn_classification/cell_2/bidirectional_rnn/bw/bw/while/basic_lstm_cell/concat/axisConstA^rnn_classification/cell_2/bidirectional_rnn/bw/bw/while/Identity*
dtype0*
_output_shapes
: *
value	B :
ń
Nrnn_classification/cell_2/bidirectional_rnn/bw/bw/while/basic_lstm_cell/concatConcatV2Irnn_classification/cell_2/bidirectional_rnn/bw/bw/while/TensorArrayReadV3Brnn_classification/cell_2/bidirectional_rnn/bw/bw/while/Identity_4Srnn_classification/cell_2/bidirectional_rnn/bw/bw/while/basic_lstm_cell/concat/axis*
T0*
N*
_output_shapes
:	
¨
Nrnn_classification/cell_2/bidirectional_rnn/bw/bw/while/basic_lstm_cell/MatMulMatMulNrnn_classification/cell_2/bidirectional_rnn/bw/bw/while/basic_lstm_cell/concatTrnn_classification/cell_2/bidirectional_rnn/bw/bw/while/basic_lstm_cell/MatMul/Enter*
T0*
_output_shapes
:	
Ů
Trnn_classification/cell_2/bidirectional_rnn/bw/bw/while/basic_lstm_cell/MatMul/EnterEnterJrnn_classification/cell_2/bidirectional_rnn/bw/basic_lstm_cell/kernel/read*
T0*
is_constant(*
parallel_iterations *U

frame_nameGErnn_classification/cell_2/bidirectional_rnn/bw/bw/while/while_context* 
_output_shapes
:

Ť
Ornn_classification/cell_2/bidirectional_rnn/bw/bw/while/basic_lstm_cell/BiasAddBiasAddNrnn_classification/cell_2/bidirectional_rnn/bw/bw/while/basic_lstm_cell/MatMulUrnn_classification/cell_2/bidirectional_rnn/bw/bw/while/basic_lstm_cell/BiasAdd/Enter*
T0*
_output_shapes
:	
Ó
Urnn_classification/cell_2/bidirectional_rnn/bw/bw/while/basic_lstm_cell/BiasAdd/EnterEnterHrnn_classification/cell_2/bidirectional_rnn/bw/basic_lstm_cell/bias/read*
parallel_iterations *U

frame_nameGErnn_classification/cell_2/bidirectional_rnn/bw/bw/while/while_context*
_output_shapes	
:*
T0*
is_constant(
Ô
Ornn_classification/cell_2/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Const_1ConstA^rnn_classification/cell_2/bidirectional_rnn/bw/bw/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
Ň
Mrnn_classification/cell_2/bidirectional_rnn/bw/bw/while/basic_lstm_cell/splitSplitMrnn_classification/cell_2/bidirectional_rnn/bw/bw/while/basic_lstm_cell/ConstOrnn_classification/cell_2/bidirectional_rnn/bw/bw/while/basic_lstm_cell/BiasAdd*
T0*
	num_split*@
_output_shapes.
,:	:	:	:	
×
Ornn_classification/cell_2/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Const_2ConstA^rnn_classification/cell_2/bidirectional_rnn/bw/bw/while/Identity*
valueB
 *  ?*
dtype0*
_output_shapes
: 

Krnn_classification/cell_2/bidirectional_rnn/bw/bw/while/basic_lstm_cell/AddAddOrnn_classification/cell_2/bidirectional_rnn/bw/bw/while/basic_lstm_cell/split:2Ornn_classification/cell_2/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Const_2*
T0*
_output_shapes
:	
Ń
Ornn_classification/cell_2/bidirectional_rnn/bw/bw/while/basic_lstm_cell/SigmoidSigmoidKrnn_classification/cell_2/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Add*
T0*
_output_shapes
:	

Krnn_classification/cell_2/bidirectional_rnn/bw/bw/while/basic_lstm_cell/MulMulBrnn_classification/cell_2/bidirectional_rnn/bw/bw/while/Identity_3Ornn_classification/cell_2/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Sigmoid*
_output_shapes
:	*
T0
Ő
Qrnn_classification/cell_2/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Sigmoid_1SigmoidMrnn_classification/cell_2/bidirectional_rnn/bw/bw/while/basic_lstm_cell/split*
T0*
_output_shapes
:	
Ď
Lrnn_classification/cell_2/bidirectional_rnn/bw/bw/while/basic_lstm_cell/TanhTanhOrnn_classification/cell_2/bidirectional_rnn/bw/bw/while/basic_lstm_cell/split:1*
T0*
_output_shapes
:	

Mrnn_classification/cell_2/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_1MulQrnn_classification/cell_2/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Sigmoid_1Lrnn_classification/cell_2/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Tanh*
T0*
_output_shapes
:	

Mrnn_classification/cell_2/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Add_1AddKrnn_classification/cell_2/bidirectional_rnn/bw/bw/while/basic_lstm_cell/MulMrnn_classification/cell_2/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_1*
T0*
_output_shapes
:	
Ď
Nrnn_classification/cell_2/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Tanh_1TanhMrnn_classification/cell_2/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Add_1*
T0*
_output_shapes
:	
×
Qrnn_classification/cell_2/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Sigmoid_2SigmoidOrnn_classification/cell_2/bidirectional_rnn/bw/bw/while/basic_lstm_cell/split:3*
T0*
_output_shapes
:	
Ą
Mrnn_classification/cell_2/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_2MulNrnn_classification/cell_2/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Tanh_1Qrnn_classification/cell_2/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Sigmoid_2*
T0*
_output_shapes
:	
Ż
>rnn_classification/cell_2/bidirectional_rnn/bw/bw/while/SelectSelectDrnn_classification/cell_2/bidirectional_rnn/bw/bw/while/GreaterEqualDrnn_classification/cell_2/bidirectional_rnn/bw/bw/while/Select/EnterMrnn_classification/cell_2/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_2*
T0*`
_classV
TRloc:@rnn_classification/cell_2/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_2*
_output_shapes
:	

Drnn_classification/cell_2/bidirectional_rnn/bw/bw/while/Select/EnterEnter7rnn_classification/cell_2/bidirectional_rnn/bw/bw/zeros*
is_constant(*
_output_shapes
:	*U

frame_nameGErnn_classification/cell_2/bidirectional_rnn/bw/bw/while/while_context*
T0*`
_classV
TRloc:@rnn_classification/cell_2/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_2*
parallel_iterations 
Ż
@rnn_classification/cell_2/bidirectional_rnn/bw/bw/while/Select_1SelectDrnn_classification/cell_2/bidirectional_rnn/bw/bw/while/GreaterEqualBrnn_classification/cell_2/bidirectional_rnn/bw/bw/while/Identity_3Mrnn_classification/cell_2/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Add_1*
T0*`
_classV
TRloc:@rnn_classification/cell_2/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Add_1*
_output_shapes
:	
Ż
@rnn_classification/cell_2/bidirectional_rnn/bw/bw/while/Select_2SelectDrnn_classification/cell_2/bidirectional_rnn/bw/bw/while/GreaterEqualBrnn_classification/cell_2/bidirectional_rnn/bw/bw/while/Identity_4Mrnn_classification/cell_2/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_2*
T0*`
_classV
TRloc:@rnn_classification/cell_2/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_2*
_output_shapes
:	

[rnn_classification/cell_2/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3TensorArrayWriteV3arnn_classification/cell_2/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3/EnterBrnn_classification/cell_2/bidirectional_rnn/bw/bw/while/Identity_1>rnn_classification/cell_2/bidirectional_rnn/bw/bw/while/SelectBrnn_classification/cell_2/bidirectional_rnn/bw/bw/while/Identity_2*
_output_shapes
: *
T0*`
_classV
TRloc:@rnn_classification/cell_2/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_2
ľ
arnn_classification/cell_2/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3/EnterEnter=rnn_classification/cell_2/bidirectional_rnn/bw/bw/TensorArray*
parallel_iterations *
is_constant(*U

frame_nameGErnn_classification/cell_2/bidirectional_rnn/bw/bw/while/while_context*
_output_shapes
:*
T0*`
_classV
TRloc:@rnn_classification/cell_2/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_2
Ä
?rnn_classification/cell_2/bidirectional_rnn/bw/bw/while/add_1/yConstA^rnn_classification/cell_2/bidirectional_rnn/bw/bw/while/Identity*
dtype0*
_output_shapes
: *
value	B :
ę
=rnn_classification/cell_2/bidirectional_rnn/bw/bw/while/add_1AddBrnn_classification/cell_2/bidirectional_rnn/bw/bw/while/Identity_1?rnn_classification/cell_2/bidirectional_rnn/bw/bw/while/add_1/y*
T0*
_output_shapes
: 
´
Ernn_classification/cell_2/bidirectional_rnn/bw/bw/while/NextIterationNextIteration;rnn_classification/cell_2/bidirectional_rnn/bw/bw/while/add*
T0*
_output_shapes
: 
¸
Grnn_classification/cell_2/bidirectional_rnn/bw/bw/while/NextIteration_1NextIteration=rnn_classification/cell_2/bidirectional_rnn/bw/bw/while/add_1*
T0*
_output_shapes
: 
Ö
Grnn_classification/cell_2/bidirectional_rnn/bw/bw/while/NextIteration_2NextIteration[rnn_classification/cell_2/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3*
_output_shapes
: *
T0
Ä
Grnn_classification/cell_2/bidirectional_rnn/bw/bw/while/NextIteration_3NextIteration@rnn_classification/cell_2/bidirectional_rnn/bw/bw/while/Select_1*
T0*
_output_shapes
:	
Ä
Grnn_classification/cell_2/bidirectional_rnn/bw/bw/while/NextIteration_4NextIteration@rnn_classification/cell_2/bidirectional_rnn/bw/bw/while/Select_2*
T0*
_output_shapes
:	
Ľ
<rnn_classification/cell_2/bidirectional_rnn/bw/bw/while/ExitExit>rnn_classification/cell_2/bidirectional_rnn/bw/bw/while/Switch*
T0*
_output_shapes
: 
Š
>rnn_classification/cell_2/bidirectional_rnn/bw/bw/while/Exit_1Exit@rnn_classification/cell_2/bidirectional_rnn/bw/bw/while/Switch_1*
T0*
_output_shapes
: 
Š
>rnn_classification/cell_2/bidirectional_rnn/bw/bw/while/Exit_2Exit@rnn_classification/cell_2/bidirectional_rnn/bw/bw/while/Switch_2*
T0*
_output_shapes
: 
˛
>rnn_classification/cell_2/bidirectional_rnn/bw/bw/while/Exit_3Exit@rnn_classification/cell_2/bidirectional_rnn/bw/bw/while/Switch_3*
T0*
_output_shapes
:	
˛
>rnn_classification/cell_2/bidirectional_rnn/bw/bw/while/Exit_4Exit@rnn_classification/cell_2/bidirectional_rnn/bw/bw/while/Switch_4*
T0*
_output_shapes
:	
Ň
Trnn_classification/cell_2/bidirectional_rnn/bw/bw/TensorArrayStack/TensorArraySizeV3TensorArraySizeV3=rnn_classification/cell_2/bidirectional_rnn/bw/bw/TensorArray>rnn_classification/cell_2/bidirectional_rnn/bw/bw/while/Exit_2*P
_classF
DBloc:@rnn_classification/cell_2/bidirectional_rnn/bw/bw/TensorArray*
_output_shapes
: 
â
Nrnn_classification/cell_2/bidirectional_rnn/bw/bw/TensorArrayStack/range/startConst*
value	B : *P
_classF
DBloc:@rnn_classification/cell_2/bidirectional_rnn/bw/bw/TensorArray*
dtype0*
_output_shapes
: 
â
Nrnn_classification/cell_2/bidirectional_rnn/bw/bw/TensorArrayStack/range/deltaConst*
value	B :*P
_classF
DBloc:@rnn_classification/cell_2/bidirectional_rnn/bw/bw/TensorArray*
dtype0*
_output_shapes
: 
ž
Hrnn_classification/cell_2/bidirectional_rnn/bw/bw/TensorArrayStack/rangeRangeNrnn_classification/cell_2/bidirectional_rnn/bw/bw/TensorArrayStack/range/startTrnn_classification/cell_2/bidirectional_rnn/bw/bw/TensorArrayStack/TensorArraySizeV3Nrnn_classification/cell_2/bidirectional_rnn/bw/bw/TensorArrayStack/range/delta*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*P
_classF
DBloc:@rnn_classification/cell_2/bidirectional_rnn/bw/bw/TensorArray
á
Vrnn_classification/cell_2/bidirectional_rnn/bw/bw/TensorArrayStack/TensorArrayGatherV3TensorArrayGatherV3=rnn_classification/cell_2/bidirectional_rnn/bw/bw/TensorArrayHrnn_classification/cell_2/bidirectional_rnn/bw/bw/TensorArrayStack/range>rnn_classification/cell_2/bidirectional_rnn/bw/bw/while/Exit_2*
dtype0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙*
element_shape:	*P
_classF
DBloc:@rnn_classification/cell_2/bidirectional_rnn/bw/bw/TensorArray

9rnn_classification/cell_2/bidirectional_rnn/bw/bw/Const_5Const*
valueB:*
dtype0*
_output_shapes
:
z
8rnn_classification/cell_2/bidirectional_rnn/bw/bw/Rank_1Const*
value	B :*
dtype0*
_output_shapes
: 

?rnn_classification/cell_2/bidirectional_rnn/bw/bw/range_1/startConst*
dtype0*
_output_shapes
: *
value	B :

?rnn_classification/cell_2/bidirectional_rnn/bw/bw/range_1/deltaConst*
value	B :*
dtype0*
_output_shapes
: 

9rnn_classification/cell_2/bidirectional_rnn/bw/bw/range_1Range?rnn_classification/cell_2/bidirectional_rnn/bw/bw/range_1/start8rnn_classification/cell_2/bidirectional_rnn/bw/bw/Rank_1?rnn_classification/cell_2/bidirectional_rnn/bw/bw/range_1/delta*
_output_shapes
:

Crnn_classification/cell_2/bidirectional_rnn/bw/bw/concat_2/values_0Const*
valueB"       *
dtype0*
_output_shapes
:

?rnn_classification/cell_2/bidirectional_rnn/bw/bw/concat_2/axisConst*
value	B : *
dtype0*
_output_shapes
: 
ľ
:rnn_classification/cell_2/bidirectional_rnn/bw/bw/concat_2ConcatV2Crnn_classification/cell_2/bidirectional_rnn/bw/bw/concat_2/values_09rnn_classification/cell_2/bidirectional_rnn/bw/bw/range_1?rnn_classification/cell_2/bidirectional_rnn/bw/bw/concat_2/axis*
T0*
N*
_output_shapes
:

=rnn_classification/cell_2/bidirectional_rnn/bw/bw/transpose_1	TransposeVrnn_classification/cell_2/bidirectional_rnn/bw/bw/TensorArrayStack/TensorArrayGatherV3:rnn_classification/cell_2/bidirectional_rnn/bw/bw/concat_2*,
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ę
)rnn_classification/cell_2/ReverseSequenceReverseSequence=rnn_classification/cell_2/bidirectional_rnn/bw/bw/transpose_1Squeeze*
T0*
seq_dim*,
_output_shapes
:˙˙˙˙˙˙˙˙˙
g
%rnn_classification/cell_2/concat/axisConst*
dtype0*
_output_shapes
: *
value	B :
ý
 rnn_classification/cell_2/concatConcatV2=rnn_classification/cell_2/bidirectional_rnn/fw/fw/transpose_1)rnn_classification/cell_2/ReverseSequence%rnn_classification/cell_2/concat/axis*
T0*
N*,
_output_shapes
:˙˙˙˙˙˙˙˙˙
U
ShapeShape rnn_classification/cell_2/concat*
T0*
_output_shapes
:
]
strided_slice/stackConst*
valueB:*
dtype0*
_output_shapes
:
_
strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
_
strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
­
strided_sliceStridedSliceShapestrided_slice/stackstrided_slice/stack_1strided_slice/stack_2*
shrink_axis_mask*
_output_shapes
: *
Index0*
T0
T
SequenceMask/ConstConst*
dtype0*
_output_shapes
: *
value	B : 
V
SequenceMask/Const_1Const*
dtype0*
_output_shapes
: *
value	B :
y
SequenceMask/RangeRangeSequenceMask/Conststrided_sliceSequenceMask/Const_1*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
f
SequenceMask/ExpandDims/dimConst*
valueB :
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
: 
t
SequenceMask/ExpandDims
ExpandDimsSqueezeSequenceMask/ExpandDims/dim*
T0	*
_output_shapes

:
j
SequenceMask/CastCastSequenceMask/ExpandDims*

SrcT0	*

DstT0*
_output_shapes

:
r
SequenceMask/LessLessSequenceMask/RangeSequenceMask/Cast*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
P
ExpandDims/dimConst*
value	B :*
dtype0*
_output_shapes
: 
q

ExpandDims
ExpandDimsSequenceMask/LessExpandDims/dim*+
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

W
Shape_1Shape rnn_classification/cell_2/concat*
_output_shapes
:*
T0
_
strided_slice_1/stackConst*
valueB:*
dtype0*
_output_shapes
:
a
strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
a
strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
ˇ
strided_slice_1StridedSliceShape_1strided_slice_1/stackstrided_slice_1/stack_1strided_slice_1/stack_2*
_output_shapes
: *
T0*
Index0*
shrink_axis_mask
R
Tile/multiples/0Const*
value	B :*
dtype0*
_output_shapes
: 
R
Tile/multiples/1Const*
value	B :*
dtype0*
_output_shapes
: 
y
Tile/multiplesPackTile/multiples/0Tile/multiples/1strided_slice_1*
T0*
N*
_output_shapes
:
g
TileTile
ExpandDimsTile/multiples*
T0
*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
p

zeros_like	ZerosLike rnn_classification/cell_2/concat*,
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
{
SelectSelectTile rnn_classification/cell_2/concat
zeros_like*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙
W
Sum/reduction_indicesConst*
dtype0*
_output_shapes
: *
value	B :
S
SumSumSelectSum/reduction_indices*
T0*
_output_shapes
:	

-dense/kernel/Initializer/random_uniform/shapeConst*
valueB"   \  *
_class
loc:@dense/kernel*
dtype0*
_output_shapes
:

+dense/kernel/Initializer/random_uniform/minConst*
valueB
 *čĚ˝*
_class
loc:@dense/kernel*
dtype0*
_output_shapes
: 

+dense/kernel/Initializer/random_uniform/maxConst*
valueB
 *čĚ=*
_class
loc:@dense/kernel*
dtype0*
_output_shapes
: 
Î
5dense/kernel/Initializer/random_uniform/RandomUniformRandomUniform-dense/kernel/Initializer/random_uniform/shape*
dtype0* 
_output_shapes
:
Ü*
T0*
_class
loc:@dense/kernel
Î
+dense/kernel/Initializer/random_uniform/subSub+dense/kernel/Initializer/random_uniform/max+dense/kernel/Initializer/random_uniform/min*
T0*
_class
loc:@dense/kernel*
_output_shapes
: 
â
+dense/kernel/Initializer/random_uniform/mulMul5dense/kernel/Initializer/random_uniform/RandomUniform+dense/kernel/Initializer/random_uniform/sub*
T0*
_class
loc:@dense/kernel* 
_output_shapes
:
Ü
Ô
'dense/kernel/Initializer/random_uniformAdd+dense/kernel/Initializer/random_uniform/mul+dense/kernel/Initializer/random_uniform/min*
T0*
_class
loc:@dense/kernel* 
_output_shapes
:
Ü

dense/kernel
VariableV2*
_class
loc:@dense/kernel*
shape:
Ü*
dtype0* 
_output_shapes
:
Ü
 
dense/kernel/AssignAssigndense/kernel'dense/kernel/Initializer/random_uniform*
T0*
_class
loc:@dense/kernel* 
_output_shapes
:
Ü
w
dense/kernel/readIdentitydense/kernel*
T0*
_class
loc:@dense/kernel* 
_output_shapes
:
Ü

dense/bias/Initializer/zerosConst*
dtype0*
_output_shapes	
:Ü*
valueBÜ*    *
_class
loc:@dense/bias
s

dense/bias
VariableV2*
shape:Ü*
dtype0*
_output_shapes	
:Ü*
_class
loc:@dense/bias

dense/bias/AssignAssign
dense/biasdense/bias/Initializer/zeros*
T0*
_class
loc:@dense/bias*
_output_shapes	
:Ü
l
dense/bias/readIdentity
dense/bias*
T0*
_class
loc:@dense/bias*
_output_shapes	
:Ü
X
dense/MatMulMatMulSumdense/kernel/read*
T0*
_output_shapes
:	Ü
a
dense/BiasAddBiasAdddense/MatMuldense/bias/read*
_output_shapes
:	Ü*
T0
R
ArgMax/dimensionConst*
value	B :*
dtype0*
_output_shapes
: 
V
ArgMaxArgMaxdense/BiasAddArgMax/dimension*
_output_shapes
:*
T0
H
CastCastArgMax*

SrcT0	*

DstT0*
_output_shapes
:
P

save/ConstConst*
dtype0*
_output_shapes
: *
valueB Bmodel

save/StringJoin/inputs_1Const*<
value3B1 B+_temp_5ba438e5f12841638fedead493fb848a/part*
dtype0*
_output_shapes
: 
d
save/StringJoin
StringJoin
save/Constsave/StringJoin/inputs_1*
N*
_output_shapes
: 
Q
save/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
k
save/ShardedFilename/shardConst"/device:CPU:0*
dtype0*
_output_shapes
: *
value	B : 

save/ShardedFilenameShardedFilenamesave/StringJoinsave/ShardedFilename/shardsave/num_shards"/device:CPU:0*
_output_shapes
: 
Ŕ
save/SaveV2/tensor_namesConst"/device:CPU:0*ä
valueÚB×Bconv1d_0/biasBconv1d_0/kernelBconv1d_1/biasBconv1d_1/kernelBconv1d_2/biasBconv1d_2/kernelB
dense/biasBdense/kernelBglobal_stepBCrnn_classification/cell_0/bidirectional_rnn/bw/basic_lstm_cell/biasBErnn_classification/cell_0/bidirectional_rnn/bw/basic_lstm_cell/kernelBCrnn_classification/cell_0/bidirectional_rnn/fw/basic_lstm_cell/biasBErnn_classification/cell_0/bidirectional_rnn/fw/basic_lstm_cell/kernelBCrnn_classification/cell_1/bidirectional_rnn/bw/basic_lstm_cell/biasBErnn_classification/cell_1/bidirectional_rnn/bw/basic_lstm_cell/kernelBCrnn_classification/cell_1/bidirectional_rnn/fw/basic_lstm_cell/biasBErnn_classification/cell_1/bidirectional_rnn/fw/basic_lstm_cell/kernelBCrnn_classification/cell_2/bidirectional_rnn/bw/basic_lstm_cell/biasBErnn_classification/cell_2/bidirectional_rnn/bw/basic_lstm_cell/kernelBCrnn_classification/cell_2/bidirectional_rnn/fw/basic_lstm_cell/biasBErnn_classification/cell_2/bidirectional_rnn/fw/basic_lstm_cell/kernel*
dtype0*
_output_shapes
:

save/SaveV2/shape_and_slicesConst"/device:CPU:0*=
value4B2B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:
ć
save/SaveV2SaveV2save/ShardedFilenamesave/SaveV2/tensor_namessave/SaveV2/shape_and_slicesconv1d_0/biasconv1d_0/kernelconv1d_1/biasconv1d_1/kernelconv1d_2/biasconv1d_2/kernel
dense/biasdense/kernelglobal_stepCrnn_classification/cell_0/bidirectional_rnn/bw/basic_lstm_cell/biasErnn_classification/cell_0/bidirectional_rnn/bw/basic_lstm_cell/kernelCrnn_classification/cell_0/bidirectional_rnn/fw/basic_lstm_cell/biasErnn_classification/cell_0/bidirectional_rnn/fw/basic_lstm_cell/kernelCrnn_classification/cell_1/bidirectional_rnn/bw/basic_lstm_cell/biasErnn_classification/cell_1/bidirectional_rnn/bw/basic_lstm_cell/kernelCrnn_classification/cell_1/bidirectional_rnn/fw/basic_lstm_cell/biasErnn_classification/cell_1/bidirectional_rnn/fw/basic_lstm_cell/kernelCrnn_classification/cell_2/bidirectional_rnn/bw/basic_lstm_cell/biasErnn_classification/cell_2/bidirectional_rnn/bw/basic_lstm_cell/kernelCrnn_classification/cell_2/bidirectional_rnn/fw/basic_lstm_cell/biasErnn_classification/cell_2/bidirectional_rnn/fw/basic_lstm_cell/kernel"/device:CPU:0*#
dtypes
2	
 
save/control_dependencyIdentitysave/ShardedFilename^save/SaveV2"/device:CPU:0*
_output_shapes
: *
T0*'
_class
loc:@save/ShardedFilename
 
+save/MergeV2Checkpoints/checkpoint_prefixesPacksave/ShardedFilename^save/control_dependency"/device:CPU:0*
N*
_output_shapes
:*
T0
u
save/MergeV2CheckpointsMergeV2Checkpoints+save/MergeV2Checkpoints/checkpoint_prefixes
save/Const"/device:CPU:0

save/IdentityIdentity
save/Const^save/MergeV2Checkpoints^save/control_dependency"/device:CPU:0*
_output_shapes
: *
T0
Ă
save/RestoreV2/tensor_namesConst"/device:CPU:0*
dtype0*
_output_shapes
:*ä
valueÚB×Bconv1d_0/biasBconv1d_0/kernelBconv1d_1/biasBconv1d_1/kernelBconv1d_2/biasBconv1d_2/kernelB
dense/biasBdense/kernelBglobal_stepBCrnn_classification/cell_0/bidirectional_rnn/bw/basic_lstm_cell/biasBErnn_classification/cell_0/bidirectional_rnn/bw/basic_lstm_cell/kernelBCrnn_classification/cell_0/bidirectional_rnn/fw/basic_lstm_cell/biasBErnn_classification/cell_0/bidirectional_rnn/fw/basic_lstm_cell/kernelBCrnn_classification/cell_1/bidirectional_rnn/bw/basic_lstm_cell/biasBErnn_classification/cell_1/bidirectional_rnn/bw/basic_lstm_cell/kernelBCrnn_classification/cell_1/bidirectional_rnn/fw/basic_lstm_cell/biasBErnn_classification/cell_1/bidirectional_rnn/fw/basic_lstm_cell/kernelBCrnn_classification/cell_2/bidirectional_rnn/bw/basic_lstm_cell/biasBErnn_classification/cell_2/bidirectional_rnn/bw/basic_lstm_cell/kernelBCrnn_classification/cell_2/bidirectional_rnn/fw/basic_lstm_cell/biasBErnn_classification/cell_2/bidirectional_rnn/fw/basic_lstm_cell/kernel

save/RestoreV2/shape_and_slicesConst"/device:CPU:0*=
value4B2B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:

save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices"/device:CPU:0*#
dtypes
2	*h
_output_shapesV
T:::::::::::::::::::::
{
save/AssignAssignconv1d_0/biassave/RestoreV2*
_output_shapes
:0*
T0* 
_class
loc:@conv1d_0/bias

save/Assign_1Assignconv1d_0/kernelsave/RestoreV2:1*"
_output_shapes
:0*
T0*"
_class
loc:@conv1d_0/kernel

save/Assign_2Assignconv1d_1/biassave/RestoreV2:2*
T0* 
_class
loc:@conv1d_1/bias*
_output_shapes
:@

save/Assign_3Assignconv1d_1/kernelsave/RestoreV2:3*
T0*"
_class
loc:@conv1d_1/kernel*"
_output_shapes
:0@

save/Assign_4Assignconv1d_2/biassave/RestoreV2:4*
_output_shapes
:`*
T0* 
_class
loc:@conv1d_2/bias

save/Assign_5Assignconv1d_2/kernelsave/RestoreV2:5*
T0*"
_class
loc:@conv1d_2/kernel*"
_output_shapes
:@`
z
save/Assign_6Assign
dense/biassave/RestoreV2:6*
T0*
_class
loc:@dense/bias*
_output_shapes	
:Ü

save/Assign_7Assigndense/kernelsave/RestoreV2:7* 
_output_shapes
:
Ü*
T0*
_class
loc:@dense/kernel
w
save/Assign_8Assignglobal_stepsave/RestoreV2:8*
_output_shapes
: *
T0	*
_class
loc:@global_step
ě
save/Assign_9AssignCrnn_classification/cell_0/bidirectional_rnn/bw/basic_lstm_cell/biassave/RestoreV2:9*
_output_shapes	
:*
T0*V
_classL
JHloc:@rnn_classification/cell_0/bidirectional_rnn/bw/basic_lstm_cell/bias
÷
save/Assign_10AssignErnn_classification/cell_0/bidirectional_rnn/bw/basic_lstm_cell/kernelsave/RestoreV2:10* 
_output_shapes
:
ŕ*
T0*X
_classN
LJloc:@rnn_classification/cell_0/bidirectional_rnn/bw/basic_lstm_cell/kernel
î
save/Assign_11AssignCrnn_classification/cell_0/bidirectional_rnn/fw/basic_lstm_cell/biassave/RestoreV2:11*
T0*V
_classL
JHloc:@rnn_classification/cell_0/bidirectional_rnn/fw/basic_lstm_cell/bias*
_output_shapes	
:
÷
save/Assign_12AssignErnn_classification/cell_0/bidirectional_rnn/fw/basic_lstm_cell/kernelsave/RestoreV2:12* 
_output_shapes
:
ŕ*
T0*X
_classN
LJloc:@rnn_classification/cell_0/bidirectional_rnn/fw/basic_lstm_cell/kernel
î
save/Assign_13AssignCrnn_classification/cell_1/bidirectional_rnn/bw/basic_lstm_cell/biassave/RestoreV2:13*
T0*V
_classL
JHloc:@rnn_classification/cell_1/bidirectional_rnn/bw/basic_lstm_cell/bias*
_output_shapes	
:
÷
save/Assign_14AssignErnn_classification/cell_1/bidirectional_rnn/bw/basic_lstm_cell/kernelsave/RestoreV2:14* 
_output_shapes
:
*
T0*X
_classN
LJloc:@rnn_classification/cell_1/bidirectional_rnn/bw/basic_lstm_cell/kernel
î
save/Assign_15AssignCrnn_classification/cell_1/bidirectional_rnn/fw/basic_lstm_cell/biassave/RestoreV2:15*
_output_shapes	
:*
T0*V
_classL
JHloc:@rnn_classification/cell_1/bidirectional_rnn/fw/basic_lstm_cell/bias
÷
save/Assign_16AssignErnn_classification/cell_1/bidirectional_rnn/fw/basic_lstm_cell/kernelsave/RestoreV2:16*
T0*X
_classN
LJloc:@rnn_classification/cell_1/bidirectional_rnn/fw/basic_lstm_cell/kernel* 
_output_shapes
:

î
save/Assign_17AssignCrnn_classification/cell_2/bidirectional_rnn/bw/basic_lstm_cell/biassave/RestoreV2:17*
T0*V
_classL
JHloc:@rnn_classification/cell_2/bidirectional_rnn/bw/basic_lstm_cell/bias*
_output_shapes	
:
÷
save/Assign_18AssignErnn_classification/cell_2/bidirectional_rnn/bw/basic_lstm_cell/kernelsave/RestoreV2:18*
T0*X
_classN
LJloc:@rnn_classification/cell_2/bidirectional_rnn/bw/basic_lstm_cell/kernel* 
_output_shapes
:

î
save/Assign_19AssignCrnn_classification/cell_2/bidirectional_rnn/fw/basic_lstm_cell/biassave/RestoreV2:19*
_output_shapes	
:*
T0*V
_classL
JHloc:@rnn_classification/cell_2/bidirectional_rnn/fw/basic_lstm_cell/bias
÷
save/Assign_20AssignErnn_classification/cell_2/bidirectional_rnn/fw/basic_lstm_cell/kernelsave/RestoreV2:20*
T0*X
_classN
LJloc:@rnn_classification/cell_2/bidirectional_rnn/fw/basic_lstm_cell/kernel* 
_output_shapes
:

ó
save/restore_shardNoOp^save/Assign^save/Assign_1^save/Assign_10^save/Assign_11^save/Assign_12^save/Assign_13^save/Assign_14^save/Assign_15^save/Assign_16^save/Assign_17^save/Assign_18^save/Assign_19^save/Assign_2^save/Assign_20^save/Assign_3^save/Assign_4^save/Assign_5^save/Assign_6^save/Assign_7^save/Assign_8^save/Assign_9
-
save/restore_allNoOp^save/restore_shard

initNoOp

init_all_tablesNoOp

init_1NoOp
4

group_depsNoOp^init^init_1^init_all_tables
R
save_1/ConstConst*
dtype0*
_output_shapes
: *
valueB Bmodel

save_1/StringJoin/inputs_1Const*<
value3B1 B+_temp_00b50151f42d400685b93db20f235fee/part*
dtype0*
_output_shapes
: 
j
save_1/StringJoin
StringJoinsave_1/Constsave_1/StringJoin/inputs_1*
N*
_output_shapes
: 
S
save_1/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
m
save_1/ShardedFilename/shardConst"/device:CPU:0*
value	B : *
dtype0*
_output_shapes
: 

save_1/ShardedFilenameShardedFilenamesave_1/StringJoinsave_1/ShardedFilename/shardsave_1/num_shards"/device:CPU:0*
_output_shapes
: 
Â
save_1/SaveV2/tensor_namesConst"/device:CPU:0*ä
valueÚB×Bconv1d_0/biasBconv1d_0/kernelBconv1d_1/biasBconv1d_1/kernelBconv1d_2/biasBconv1d_2/kernelB
dense/biasBdense/kernelBglobal_stepBCrnn_classification/cell_0/bidirectional_rnn/bw/basic_lstm_cell/biasBErnn_classification/cell_0/bidirectional_rnn/bw/basic_lstm_cell/kernelBCrnn_classification/cell_0/bidirectional_rnn/fw/basic_lstm_cell/biasBErnn_classification/cell_0/bidirectional_rnn/fw/basic_lstm_cell/kernelBCrnn_classification/cell_1/bidirectional_rnn/bw/basic_lstm_cell/biasBErnn_classification/cell_1/bidirectional_rnn/bw/basic_lstm_cell/kernelBCrnn_classification/cell_1/bidirectional_rnn/fw/basic_lstm_cell/biasBErnn_classification/cell_1/bidirectional_rnn/fw/basic_lstm_cell/kernelBCrnn_classification/cell_2/bidirectional_rnn/bw/basic_lstm_cell/biasBErnn_classification/cell_2/bidirectional_rnn/bw/basic_lstm_cell/kernelBCrnn_classification/cell_2/bidirectional_rnn/fw/basic_lstm_cell/biasBErnn_classification/cell_2/bidirectional_rnn/fw/basic_lstm_cell/kernel*
dtype0*
_output_shapes
:

save_1/SaveV2/shape_and_slicesConst"/device:CPU:0*=
value4B2B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:
î
save_1/SaveV2SaveV2save_1/ShardedFilenamesave_1/SaveV2/tensor_namessave_1/SaveV2/shape_and_slicesconv1d_0/biasconv1d_0/kernelconv1d_1/biasconv1d_1/kernelconv1d_2/biasconv1d_2/kernel
dense/biasdense/kernelglobal_stepCrnn_classification/cell_0/bidirectional_rnn/bw/basic_lstm_cell/biasErnn_classification/cell_0/bidirectional_rnn/bw/basic_lstm_cell/kernelCrnn_classification/cell_0/bidirectional_rnn/fw/basic_lstm_cell/biasErnn_classification/cell_0/bidirectional_rnn/fw/basic_lstm_cell/kernelCrnn_classification/cell_1/bidirectional_rnn/bw/basic_lstm_cell/biasErnn_classification/cell_1/bidirectional_rnn/bw/basic_lstm_cell/kernelCrnn_classification/cell_1/bidirectional_rnn/fw/basic_lstm_cell/biasErnn_classification/cell_1/bidirectional_rnn/fw/basic_lstm_cell/kernelCrnn_classification/cell_2/bidirectional_rnn/bw/basic_lstm_cell/biasErnn_classification/cell_2/bidirectional_rnn/bw/basic_lstm_cell/kernelCrnn_classification/cell_2/bidirectional_rnn/fw/basic_lstm_cell/biasErnn_classification/cell_2/bidirectional_rnn/fw/basic_lstm_cell/kernel"/device:CPU:0*#
dtypes
2	
¨
save_1/control_dependencyIdentitysave_1/ShardedFilename^save_1/SaveV2"/device:CPU:0*
T0*)
_class
loc:@save_1/ShardedFilename*
_output_shapes
: 
Ś
-save_1/MergeV2Checkpoints/checkpoint_prefixesPacksave_1/ShardedFilename^save_1/control_dependency"/device:CPU:0*
N*
_output_shapes
:*
T0
{
save_1/MergeV2CheckpointsMergeV2Checkpoints-save_1/MergeV2Checkpoints/checkpoint_prefixessave_1/Const"/device:CPU:0

save_1/IdentityIdentitysave_1/Const^save_1/MergeV2Checkpoints^save_1/control_dependency"/device:CPU:0*
_output_shapes
: *
T0
Ĺ
save_1/RestoreV2/tensor_namesConst"/device:CPU:0*ä
valueÚB×Bconv1d_0/biasBconv1d_0/kernelBconv1d_1/biasBconv1d_1/kernelBconv1d_2/biasBconv1d_2/kernelB
dense/biasBdense/kernelBglobal_stepBCrnn_classification/cell_0/bidirectional_rnn/bw/basic_lstm_cell/biasBErnn_classification/cell_0/bidirectional_rnn/bw/basic_lstm_cell/kernelBCrnn_classification/cell_0/bidirectional_rnn/fw/basic_lstm_cell/biasBErnn_classification/cell_0/bidirectional_rnn/fw/basic_lstm_cell/kernelBCrnn_classification/cell_1/bidirectional_rnn/bw/basic_lstm_cell/biasBErnn_classification/cell_1/bidirectional_rnn/bw/basic_lstm_cell/kernelBCrnn_classification/cell_1/bidirectional_rnn/fw/basic_lstm_cell/biasBErnn_classification/cell_1/bidirectional_rnn/fw/basic_lstm_cell/kernelBCrnn_classification/cell_2/bidirectional_rnn/bw/basic_lstm_cell/biasBErnn_classification/cell_2/bidirectional_rnn/bw/basic_lstm_cell/kernelBCrnn_classification/cell_2/bidirectional_rnn/fw/basic_lstm_cell/biasBErnn_classification/cell_2/bidirectional_rnn/fw/basic_lstm_cell/kernel*
dtype0*
_output_shapes
:
Ą
!save_1/RestoreV2/shape_and_slicesConst"/device:CPU:0*
dtype0*
_output_shapes
:*=
value4B2B B B B B B B B B B B B B B B B B B B B B 

save_1/RestoreV2	RestoreV2save_1/Constsave_1/RestoreV2/tensor_names!save_1/RestoreV2/shape_and_slices"/device:CPU:0*#
dtypes
2	*h
_output_shapesV
T:::::::::::::::::::::

save_1/AssignAssignconv1d_0/biassave_1/RestoreV2*
T0* 
_class
loc:@conv1d_0/bias*
_output_shapes
:0

save_1/Assign_1Assignconv1d_0/kernelsave_1/RestoreV2:1*
T0*"
_class
loc:@conv1d_0/kernel*"
_output_shapes
:0

save_1/Assign_2Assignconv1d_1/biassave_1/RestoreV2:2*
T0* 
_class
loc:@conv1d_1/bias*
_output_shapes
:@

save_1/Assign_3Assignconv1d_1/kernelsave_1/RestoreV2:3*
T0*"
_class
loc:@conv1d_1/kernel*"
_output_shapes
:0@

save_1/Assign_4Assignconv1d_2/biassave_1/RestoreV2:4*
_output_shapes
:`*
T0* 
_class
loc:@conv1d_2/bias

save_1/Assign_5Assignconv1d_2/kernelsave_1/RestoreV2:5*
T0*"
_class
loc:@conv1d_2/kernel*"
_output_shapes
:@`
~
save_1/Assign_6Assign
dense/biassave_1/RestoreV2:6*
T0*
_class
loc:@dense/bias*
_output_shapes	
:Ü

save_1/Assign_7Assigndense/kernelsave_1/RestoreV2:7*
T0*
_class
loc:@dense/kernel* 
_output_shapes
:
Ü
{
save_1/Assign_8Assignglobal_stepsave_1/RestoreV2:8*
T0	*
_class
loc:@global_step*
_output_shapes
: 
đ
save_1/Assign_9AssignCrnn_classification/cell_0/bidirectional_rnn/bw/basic_lstm_cell/biassave_1/RestoreV2:9*
T0*V
_classL
JHloc:@rnn_classification/cell_0/bidirectional_rnn/bw/basic_lstm_cell/bias*
_output_shapes	
:
ű
save_1/Assign_10AssignErnn_classification/cell_0/bidirectional_rnn/bw/basic_lstm_cell/kernelsave_1/RestoreV2:10*
T0*X
_classN
LJloc:@rnn_classification/cell_0/bidirectional_rnn/bw/basic_lstm_cell/kernel* 
_output_shapes
:
ŕ
ň
save_1/Assign_11AssignCrnn_classification/cell_0/bidirectional_rnn/fw/basic_lstm_cell/biassave_1/RestoreV2:11*
T0*V
_classL
JHloc:@rnn_classification/cell_0/bidirectional_rnn/fw/basic_lstm_cell/bias*
_output_shapes	
:
ű
save_1/Assign_12AssignErnn_classification/cell_0/bidirectional_rnn/fw/basic_lstm_cell/kernelsave_1/RestoreV2:12*
T0*X
_classN
LJloc:@rnn_classification/cell_0/bidirectional_rnn/fw/basic_lstm_cell/kernel* 
_output_shapes
:
ŕ
ň
save_1/Assign_13AssignCrnn_classification/cell_1/bidirectional_rnn/bw/basic_lstm_cell/biassave_1/RestoreV2:13*
T0*V
_classL
JHloc:@rnn_classification/cell_1/bidirectional_rnn/bw/basic_lstm_cell/bias*
_output_shapes	
:
ű
save_1/Assign_14AssignErnn_classification/cell_1/bidirectional_rnn/bw/basic_lstm_cell/kernelsave_1/RestoreV2:14* 
_output_shapes
:
*
T0*X
_classN
LJloc:@rnn_classification/cell_1/bidirectional_rnn/bw/basic_lstm_cell/kernel
ň
save_1/Assign_15AssignCrnn_classification/cell_1/bidirectional_rnn/fw/basic_lstm_cell/biassave_1/RestoreV2:15*
T0*V
_classL
JHloc:@rnn_classification/cell_1/bidirectional_rnn/fw/basic_lstm_cell/bias*
_output_shapes	
:
ű
save_1/Assign_16AssignErnn_classification/cell_1/bidirectional_rnn/fw/basic_lstm_cell/kernelsave_1/RestoreV2:16*
T0*X
_classN
LJloc:@rnn_classification/cell_1/bidirectional_rnn/fw/basic_lstm_cell/kernel* 
_output_shapes
:

ň
save_1/Assign_17AssignCrnn_classification/cell_2/bidirectional_rnn/bw/basic_lstm_cell/biassave_1/RestoreV2:17*
_output_shapes	
:*
T0*V
_classL
JHloc:@rnn_classification/cell_2/bidirectional_rnn/bw/basic_lstm_cell/bias
ű
save_1/Assign_18AssignErnn_classification/cell_2/bidirectional_rnn/bw/basic_lstm_cell/kernelsave_1/RestoreV2:18*
T0*X
_classN
LJloc:@rnn_classification/cell_2/bidirectional_rnn/bw/basic_lstm_cell/kernel* 
_output_shapes
:

ň
save_1/Assign_19AssignCrnn_classification/cell_2/bidirectional_rnn/fw/basic_lstm_cell/biassave_1/RestoreV2:19*
T0*V
_classL
JHloc:@rnn_classification/cell_2/bidirectional_rnn/fw/basic_lstm_cell/bias*
_output_shapes	
:
ű
save_1/Assign_20AssignErnn_classification/cell_2/bidirectional_rnn/fw/basic_lstm_cell/kernelsave_1/RestoreV2:20*
T0*X
_classN
LJloc:@rnn_classification/cell_2/bidirectional_rnn/fw/basic_lstm_cell/kernel* 
_output_shapes
:


save_1/restore_shardNoOp^save_1/Assign^save_1/Assign_1^save_1/Assign_10^save_1/Assign_11^save_1/Assign_12^save_1/Assign_13^save_1/Assign_14^save_1/Assign_15^save_1/Assign_16^save_1/Assign_17^save_1/Assign_18^save_1/Assign_19^save_1/Assign_2^save_1/Assign_20^save_1/Assign_3^save_1/Assign_4^save_1/Assign_5^save_1/Assign_6^save_1/Assign_7^save_1/Assign_8^save_1/Assign_9
1
save_1/restore_allNoOp^save_1/restore_shard"B
save_1/Const:0save_1/Identity:0save_1/restore_all (5 @F8" 
legacy_init_op


group_deps"%
trainable_variablesé$ć$
q
conv1d_0/kernel:0conv1d_0/kernel/Assignconv1d_0/kernel/read:02,conv1d_0/kernel/Initializer/random_uniform:0
`
conv1d_0/bias:0conv1d_0/bias/Assignconv1d_0/bias/read:02!conv1d_0/bias/Initializer/zeros:0
q
conv1d_1/kernel:0conv1d_1/kernel/Assignconv1d_1/kernel/read:02,conv1d_1/kernel/Initializer/random_uniform:0
`
conv1d_1/bias:0conv1d_1/bias/Assignconv1d_1/bias/read:02!conv1d_1/bias/Initializer/zeros:0
q
conv1d_2/kernel:0conv1d_2/kernel/Assignconv1d_2/kernel/read:02,conv1d_2/kernel/Initializer/random_uniform:0
`
conv1d_2/bias:0conv1d_2/bias/Assignconv1d_2/bias/read:02!conv1d_2/bias/Initializer/zeros:0
É
Grnn_classification/cell_0/bidirectional_rnn/fw/basic_lstm_cell/kernel:0Lrnn_classification/cell_0/bidirectional_rnn/fw/basic_lstm_cell/kernel/AssignLrnn_classification/cell_0/bidirectional_rnn/fw/basic_lstm_cell/kernel/read:02brnn_classification/cell_0/bidirectional_rnn/fw/basic_lstm_cell/kernel/Initializer/random_uniform:0
¸
Ernn_classification/cell_0/bidirectional_rnn/fw/basic_lstm_cell/bias:0Jrnn_classification/cell_0/bidirectional_rnn/fw/basic_lstm_cell/bias/AssignJrnn_classification/cell_0/bidirectional_rnn/fw/basic_lstm_cell/bias/read:02Wrnn_classification/cell_0/bidirectional_rnn/fw/basic_lstm_cell/bias/Initializer/zeros:0
É
Grnn_classification/cell_0/bidirectional_rnn/bw/basic_lstm_cell/kernel:0Lrnn_classification/cell_0/bidirectional_rnn/bw/basic_lstm_cell/kernel/AssignLrnn_classification/cell_0/bidirectional_rnn/bw/basic_lstm_cell/kernel/read:02brnn_classification/cell_0/bidirectional_rnn/bw/basic_lstm_cell/kernel/Initializer/random_uniform:0
¸
Ernn_classification/cell_0/bidirectional_rnn/bw/basic_lstm_cell/bias:0Jrnn_classification/cell_0/bidirectional_rnn/bw/basic_lstm_cell/bias/AssignJrnn_classification/cell_0/bidirectional_rnn/bw/basic_lstm_cell/bias/read:02Wrnn_classification/cell_0/bidirectional_rnn/bw/basic_lstm_cell/bias/Initializer/zeros:0
É
Grnn_classification/cell_1/bidirectional_rnn/fw/basic_lstm_cell/kernel:0Lrnn_classification/cell_1/bidirectional_rnn/fw/basic_lstm_cell/kernel/AssignLrnn_classification/cell_1/bidirectional_rnn/fw/basic_lstm_cell/kernel/read:02brnn_classification/cell_1/bidirectional_rnn/fw/basic_lstm_cell/kernel/Initializer/random_uniform:0
¸
Ernn_classification/cell_1/bidirectional_rnn/fw/basic_lstm_cell/bias:0Jrnn_classification/cell_1/bidirectional_rnn/fw/basic_lstm_cell/bias/AssignJrnn_classification/cell_1/bidirectional_rnn/fw/basic_lstm_cell/bias/read:02Wrnn_classification/cell_1/bidirectional_rnn/fw/basic_lstm_cell/bias/Initializer/zeros:0
É
Grnn_classification/cell_1/bidirectional_rnn/bw/basic_lstm_cell/kernel:0Lrnn_classification/cell_1/bidirectional_rnn/bw/basic_lstm_cell/kernel/AssignLrnn_classification/cell_1/bidirectional_rnn/bw/basic_lstm_cell/kernel/read:02brnn_classification/cell_1/bidirectional_rnn/bw/basic_lstm_cell/kernel/Initializer/random_uniform:0
¸
Ernn_classification/cell_1/bidirectional_rnn/bw/basic_lstm_cell/bias:0Jrnn_classification/cell_1/bidirectional_rnn/bw/basic_lstm_cell/bias/AssignJrnn_classification/cell_1/bidirectional_rnn/bw/basic_lstm_cell/bias/read:02Wrnn_classification/cell_1/bidirectional_rnn/bw/basic_lstm_cell/bias/Initializer/zeros:0
É
Grnn_classification/cell_2/bidirectional_rnn/fw/basic_lstm_cell/kernel:0Lrnn_classification/cell_2/bidirectional_rnn/fw/basic_lstm_cell/kernel/AssignLrnn_classification/cell_2/bidirectional_rnn/fw/basic_lstm_cell/kernel/read:02brnn_classification/cell_2/bidirectional_rnn/fw/basic_lstm_cell/kernel/Initializer/random_uniform:0
¸
Ernn_classification/cell_2/bidirectional_rnn/fw/basic_lstm_cell/bias:0Jrnn_classification/cell_2/bidirectional_rnn/fw/basic_lstm_cell/bias/AssignJrnn_classification/cell_2/bidirectional_rnn/fw/basic_lstm_cell/bias/read:02Wrnn_classification/cell_2/bidirectional_rnn/fw/basic_lstm_cell/bias/Initializer/zeros:0
É
Grnn_classification/cell_2/bidirectional_rnn/bw/basic_lstm_cell/kernel:0Lrnn_classification/cell_2/bidirectional_rnn/bw/basic_lstm_cell/kernel/AssignLrnn_classification/cell_2/bidirectional_rnn/bw/basic_lstm_cell/kernel/read:02brnn_classification/cell_2/bidirectional_rnn/bw/basic_lstm_cell/kernel/Initializer/random_uniform:0
¸
Ernn_classification/cell_2/bidirectional_rnn/bw/basic_lstm_cell/bias:0Jrnn_classification/cell_2/bidirectional_rnn/bw/basic_lstm_cell/bias/AssignJrnn_classification/cell_2/bidirectional_rnn/bw/basic_lstm_cell/bias/read:02Wrnn_classification/cell_2/bidirectional_rnn/bw/basic_lstm_cell/bias/Initializer/zeros:0
e
dense/kernel:0dense/kernel/Assigndense/kernel/read:02)dense/kernel/Initializer/random_uniform:0
T
dense/bias:0dense/bias/Assigndense/bias/read:02dense/bias/Initializer/zeros:0"k
global_step\Z
X
global_step:0global_step/Assignglobal_step/read:02global_step/Initializer/zeros:0"÷°
while_contextä°ŕ°
H
Ernn_classification/cell_0/bidirectional_rnn/fw/fw/while/while_context *Brnn_classification/cell_0/bidirectional_rnn/fw/fw/while/LoopCond:02?rnn_classification/cell_0/bidirectional_rnn/fw/fw/while/Merge:0:Brnn_classification/cell_0/bidirectional_rnn/fw/fw/while/Identity:0B>rnn_classification/cell_0/bidirectional_rnn/fw/fw/while/Exit:0B@rnn_classification/cell_0/bidirectional_rnn/fw/fw/while/Exit_1:0B@rnn_classification/cell_0/bidirectional_rnn/fw/fw/while/Exit_2:0B@rnn_classification/cell_0/bidirectional_rnn/fw/fw/while/Exit_3:0B@rnn_classification/cell_0/bidirectional_rnn/fw/fw/while/Exit_4:0J@
Jrnn_classification/cell_0/bidirectional_rnn/fw/basic_lstm_cell/bias/read:0
Lrnn_classification/cell_0/bidirectional_rnn/fw/basic_lstm_cell/kernel/read:0
?rnn_classification/cell_0/bidirectional_rnn/fw/fw/CheckSeqLen:0
;rnn_classification/cell_0/bidirectional_rnn/fw/fw/Minimum:0
?rnn_classification/cell_0/bidirectional_rnn/fw/fw/TensorArray:0
nrnn_classification/cell_0/bidirectional_rnn/fw/fw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3:0
Arnn_classification/cell_0/bidirectional_rnn/fw/fw/TensorArray_1:0
Arnn_classification/cell_0/bidirectional_rnn/fw/fw/strided_slice:0
?rnn_classification/cell_0/bidirectional_rnn/fw/fw/while/Enter:0
Arnn_classification/cell_0/bidirectional_rnn/fw/fw/while/Enter_1:0
Arnn_classification/cell_0/bidirectional_rnn/fw/fw/while/Enter_2:0
Arnn_classification/cell_0/bidirectional_rnn/fw/fw/while/Enter_3:0
Arnn_classification/cell_0/bidirectional_rnn/fw/fw/while/Enter_4:0
>rnn_classification/cell_0/bidirectional_rnn/fw/fw/while/Exit:0
@rnn_classification/cell_0/bidirectional_rnn/fw/fw/while/Exit_1:0
@rnn_classification/cell_0/bidirectional_rnn/fw/fw/while/Exit_2:0
@rnn_classification/cell_0/bidirectional_rnn/fw/fw/while/Exit_3:0
@rnn_classification/cell_0/bidirectional_rnn/fw/fw/while/Exit_4:0
Lrnn_classification/cell_0/bidirectional_rnn/fw/fw/while/GreaterEqual/Enter:0
Frnn_classification/cell_0/bidirectional_rnn/fw/fw/while/GreaterEqual:0
Brnn_classification/cell_0/bidirectional_rnn/fw/fw/while/Identity:0
Drnn_classification/cell_0/bidirectional_rnn/fw/fw/while/Identity_1:0
Drnn_classification/cell_0/bidirectional_rnn/fw/fw/while/Identity_2:0
Drnn_classification/cell_0/bidirectional_rnn/fw/fw/while/Identity_3:0
Drnn_classification/cell_0/bidirectional_rnn/fw/fw/while/Identity_4:0
Drnn_classification/cell_0/bidirectional_rnn/fw/fw/while/Less/Enter:0
>rnn_classification/cell_0/bidirectional_rnn/fw/fw/while/Less:0
Frnn_classification/cell_0/bidirectional_rnn/fw/fw/while/Less_1/Enter:0
@rnn_classification/cell_0/bidirectional_rnn/fw/fw/while/Less_1:0
Drnn_classification/cell_0/bidirectional_rnn/fw/fw/while/LogicalAnd:0
Brnn_classification/cell_0/bidirectional_rnn/fw/fw/while/LoopCond:0
?rnn_classification/cell_0/bidirectional_rnn/fw/fw/while/Merge:0
?rnn_classification/cell_0/bidirectional_rnn/fw/fw/while/Merge:1
Arnn_classification/cell_0/bidirectional_rnn/fw/fw/while/Merge_1:0
Arnn_classification/cell_0/bidirectional_rnn/fw/fw/while/Merge_1:1
Arnn_classification/cell_0/bidirectional_rnn/fw/fw/while/Merge_2:0
Arnn_classification/cell_0/bidirectional_rnn/fw/fw/while/Merge_2:1
Arnn_classification/cell_0/bidirectional_rnn/fw/fw/while/Merge_3:0
Arnn_classification/cell_0/bidirectional_rnn/fw/fw/while/Merge_3:1
Arnn_classification/cell_0/bidirectional_rnn/fw/fw/while/Merge_4:0
Arnn_classification/cell_0/bidirectional_rnn/fw/fw/while/Merge_4:1
Grnn_classification/cell_0/bidirectional_rnn/fw/fw/while/NextIteration:0
Irnn_classification/cell_0/bidirectional_rnn/fw/fw/while/NextIteration_1:0
Irnn_classification/cell_0/bidirectional_rnn/fw/fw/while/NextIteration_2:0
Irnn_classification/cell_0/bidirectional_rnn/fw/fw/while/NextIteration_3:0
Irnn_classification/cell_0/bidirectional_rnn/fw/fw/while/NextIteration_4:0
Frnn_classification/cell_0/bidirectional_rnn/fw/fw/while/Select/Enter:0
@rnn_classification/cell_0/bidirectional_rnn/fw/fw/while/Select:0
Brnn_classification/cell_0/bidirectional_rnn/fw/fw/while/Select_1:0
Brnn_classification/cell_0/bidirectional_rnn/fw/fw/while/Select_2:0
@rnn_classification/cell_0/bidirectional_rnn/fw/fw/while/Switch:0
@rnn_classification/cell_0/bidirectional_rnn/fw/fw/while/Switch:1
Brnn_classification/cell_0/bidirectional_rnn/fw/fw/while/Switch_1:0
Brnn_classification/cell_0/bidirectional_rnn/fw/fw/while/Switch_1:1
Brnn_classification/cell_0/bidirectional_rnn/fw/fw/while/Switch_2:0
Brnn_classification/cell_0/bidirectional_rnn/fw/fw/while/Switch_2:1
Brnn_classification/cell_0/bidirectional_rnn/fw/fw/while/Switch_3:0
Brnn_classification/cell_0/bidirectional_rnn/fw/fw/while/Switch_3:1
Brnn_classification/cell_0/bidirectional_rnn/fw/fw/while/Switch_4:0
Brnn_classification/cell_0/bidirectional_rnn/fw/fw/while/Switch_4:1
Qrnn_classification/cell_0/bidirectional_rnn/fw/fw/while/TensorArrayReadV3/Enter:0
Srnn_classification/cell_0/bidirectional_rnn/fw/fw/while/TensorArrayReadV3/Enter_1:0
Krnn_classification/cell_0/bidirectional_rnn/fw/fw/while/TensorArrayReadV3:0
crnn_classification/cell_0/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3/Enter:0
]rnn_classification/cell_0/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3:0
?rnn_classification/cell_0/bidirectional_rnn/fw/fw/while/add/y:0
=rnn_classification/cell_0/bidirectional_rnn/fw/fw/while/add:0
Arnn_classification/cell_0/bidirectional_rnn/fw/fw/while/add_1/y:0
?rnn_classification/cell_0/bidirectional_rnn/fw/fw/while/add_1:0
Mrnn_classification/cell_0/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Add:0
Ornn_classification/cell_0/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Add_1:0
Wrnn_classification/cell_0/bidirectional_rnn/fw/fw/while/basic_lstm_cell/BiasAdd/Enter:0
Qrnn_classification/cell_0/bidirectional_rnn/fw/fw/while/basic_lstm_cell/BiasAdd:0
Ornn_classification/cell_0/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Const:0
Qrnn_classification/cell_0/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Const_1:0
Qrnn_classification/cell_0/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Const_2:0
Vrnn_classification/cell_0/bidirectional_rnn/fw/fw/while/basic_lstm_cell/MatMul/Enter:0
Prnn_classification/cell_0/bidirectional_rnn/fw/fw/while/basic_lstm_cell/MatMul:0
Mrnn_classification/cell_0/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul:0
Ornn_classification/cell_0/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_1:0
Ornn_classification/cell_0/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_2:0
Qrnn_classification/cell_0/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Sigmoid:0
Srnn_classification/cell_0/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Sigmoid_1:0
Srnn_classification/cell_0/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Sigmoid_2:0
Nrnn_classification/cell_0/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Tanh:0
Prnn_classification/cell_0/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Tanh_1:0
Urnn_classification/cell_0/bidirectional_rnn/fw/fw/while/basic_lstm_cell/concat/axis:0
Prnn_classification/cell_0/bidirectional_rnn/fw/fw/while/basic_lstm_cell/concat:0
Ornn_classification/cell_0/bidirectional_rnn/fw/fw/while/basic_lstm_cell/split:0
Ornn_classification/cell_0/bidirectional_rnn/fw/fw/while/basic_lstm_cell/split:1
Ornn_classification/cell_0/bidirectional_rnn/fw/fw/while/basic_lstm_cell/split:2
Ornn_classification/cell_0/bidirectional_rnn/fw/fw/while/basic_lstm_cell/split:3
9rnn_classification/cell_0/bidirectional_rnn/fw/fw/zeros:0Ś
Lrnn_classification/cell_0/bidirectional_rnn/fw/basic_lstm_cell/kernel/read:0Vrnn_classification/cell_0/bidirectional_rnn/fw/fw/while/basic_lstm_cell/MatMul/Enter:0Ľ
Jrnn_classification/cell_0/bidirectional_rnn/fw/basic_lstm_cell/bias/read:0Wrnn_classification/cell_0/bidirectional_rnn/fw/fw/while/basic_lstm_cell/BiasAdd/Enter:0
Arnn_classification/cell_0/bidirectional_rnn/fw/fw/strided_slice:0Drnn_classification/cell_0/bidirectional_rnn/fw/fw/while/Less/Enter:0
?rnn_classification/cell_0/bidirectional_rnn/fw/fw/CheckSeqLen:0Lrnn_classification/cell_0/bidirectional_rnn/fw/fw/while/GreaterEqual/Enter:0
9rnn_classification/cell_0/bidirectional_rnn/fw/fw/zeros:0Frnn_classification/cell_0/bidirectional_rnn/fw/fw/while/Select/Enter:0Ś
?rnn_classification/cell_0/bidirectional_rnn/fw/fw/TensorArray:0crnn_classification/cell_0/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3/Enter:0
;rnn_classification/cell_0/bidirectional_rnn/fw/fw/Minimum:0Frnn_classification/cell_0/bidirectional_rnn/fw/fw/while/Less_1/Enter:0
Arnn_classification/cell_0/bidirectional_rnn/fw/fw/TensorArray_1:0Qrnn_classification/cell_0/bidirectional_rnn/fw/fw/while/TensorArrayReadV3/Enter:0Ĺ
nrnn_classification/cell_0/bidirectional_rnn/fw/fw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3:0Srnn_classification/cell_0/bidirectional_rnn/fw/fw/while/TensorArrayReadV3/Enter_1:0R?rnn_classification/cell_0/bidirectional_rnn/fw/fw/while/Enter:0RArnn_classification/cell_0/bidirectional_rnn/fw/fw/while/Enter_1:0RArnn_classification/cell_0/bidirectional_rnn/fw/fw/while/Enter_2:0RArnn_classification/cell_0/bidirectional_rnn/fw/fw/while/Enter_3:0RArnn_classification/cell_0/bidirectional_rnn/fw/fw/while/Enter_4:0ZArnn_classification/cell_0/bidirectional_rnn/fw/fw/strided_slice:0
H
Ernn_classification/cell_0/bidirectional_rnn/bw/bw/while/while_context *Brnn_classification/cell_0/bidirectional_rnn/bw/bw/while/LoopCond:02?rnn_classification/cell_0/bidirectional_rnn/bw/bw/while/Merge:0:Brnn_classification/cell_0/bidirectional_rnn/bw/bw/while/Identity:0B>rnn_classification/cell_0/bidirectional_rnn/bw/bw/while/Exit:0B@rnn_classification/cell_0/bidirectional_rnn/bw/bw/while/Exit_1:0B@rnn_classification/cell_0/bidirectional_rnn/bw/bw/while/Exit_2:0B@rnn_classification/cell_0/bidirectional_rnn/bw/bw/while/Exit_3:0B@rnn_classification/cell_0/bidirectional_rnn/bw/bw/while/Exit_4:0J@
Jrnn_classification/cell_0/bidirectional_rnn/bw/basic_lstm_cell/bias/read:0
Lrnn_classification/cell_0/bidirectional_rnn/bw/basic_lstm_cell/kernel/read:0
?rnn_classification/cell_0/bidirectional_rnn/bw/bw/CheckSeqLen:0
;rnn_classification/cell_0/bidirectional_rnn/bw/bw/Minimum:0
?rnn_classification/cell_0/bidirectional_rnn/bw/bw/TensorArray:0
nrnn_classification/cell_0/bidirectional_rnn/bw/bw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3:0
Arnn_classification/cell_0/bidirectional_rnn/bw/bw/TensorArray_1:0
Arnn_classification/cell_0/bidirectional_rnn/bw/bw/strided_slice:0
?rnn_classification/cell_0/bidirectional_rnn/bw/bw/while/Enter:0
Arnn_classification/cell_0/bidirectional_rnn/bw/bw/while/Enter_1:0
Arnn_classification/cell_0/bidirectional_rnn/bw/bw/while/Enter_2:0
Arnn_classification/cell_0/bidirectional_rnn/bw/bw/while/Enter_3:0
Arnn_classification/cell_0/bidirectional_rnn/bw/bw/while/Enter_4:0
>rnn_classification/cell_0/bidirectional_rnn/bw/bw/while/Exit:0
@rnn_classification/cell_0/bidirectional_rnn/bw/bw/while/Exit_1:0
@rnn_classification/cell_0/bidirectional_rnn/bw/bw/while/Exit_2:0
@rnn_classification/cell_0/bidirectional_rnn/bw/bw/while/Exit_3:0
@rnn_classification/cell_0/bidirectional_rnn/bw/bw/while/Exit_4:0
Lrnn_classification/cell_0/bidirectional_rnn/bw/bw/while/GreaterEqual/Enter:0
Frnn_classification/cell_0/bidirectional_rnn/bw/bw/while/GreaterEqual:0
Brnn_classification/cell_0/bidirectional_rnn/bw/bw/while/Identity:0
Drnn_classification/cell_0/bidirectional_rnn/bw/bw/while/Identity_1:0
Drnn_classification/cell_0/bidirectional_rnn/bw/bw/while/Identity_2:0
Drnn_classification/cell_0/bidirectional_rnn/bw/bw/while/Identity_3:0
Drnn_classification/cell_0/bidirectional_rnn/bw/bw/while/Identity_4:0
Drnn_classification/cell_0/bidirectional_rnn/bw/bw/while/Less/Enter:0
>rnn_classification/cell_0/bidirectional_rnn/bw/bw/while/Less:0
Frnn_classification/cell_0/bidirectional_rnn/bw/bw/while/Less_1/Enter:0
@rnn_classification/cell_0/bidirectional_rnn/bw/bw/while/Less_1:0
Drnn_classification/cell_0/bidirectional_rnn/bw/bw/while/LogicalAnd:0
Brnn_classification/cell_0/bidirectional_rnn/bw/bw/while/LoopCond:0
?rnn_classification/cell_0/bidirectional_rnn/bw/bw/while/Merge:0
?rnn_classification/cell_0/bidirectional_rnn/bw/bw/while/Merge:1
Arnn_classification/cell_0/bidirectional_rnn/bw/bw/while/Merge_1:0
Arnn_classification/cell_0/bidirectional_rnn/bw/bw/while/Merge_1:1
Arnn_classification/cell_0/bidirectional_rnn/bw/bw/while/Merge_2:0
Arnn_classification/cell_0/bidirectional_rnn/bw/bw/while/Merge_2:1
Arnn_classification/cell_0/bidirectional_rnn/bw/bw/while/Merge_3:0
Arnn_classification/cell_0/bidirectional_rnn/bw/bw/while/Merge_3:1
Arnn_classification/cell_0/bidirectional_rnn/bw/bw/while/Merge_4:0
Arnn_classification/cell_0/bidirectional_rnn/bw/bw/while/Merge_4:1
Grnn_classification/cell_0/bidirectional_rnn/bw/bw/while/NextIteration:0
Irnn_classification/cell_0/bidirectional_rnn/bw/bw/while/NextIteration_1:0
Irnn_classification/cell_0/bidirectional_rnn/bw/bw/while/NextIteration_2:0
Irnn_classification/cell_0/bidirectional_rnn/bw/bw/while/NextIteration_3:0
Irnn_classification/cell_0/bidirectional_rnn/bw/bw/while/NextIteration_4:0
Frnn_classification/cell_0/bidirectional_rnn/bw/bw/while/Select/Enter:0
@rnn_classification/cell_0/bidirectional_rnn/bw/bw/while/Select:0
Brnn_classification/cell_0/bidirectional_rnn/bw/bw/while/Select_1:0
Brnn_classification/cell_0/bidirectional_rnn/bw/bw/while/Select_2:0
@rnn_classification/cell_0/bidirectional_rnn/bw/bw/while/Switch:0
@rnn_classification/cell_0/bidirectional_rnn/bw/bw/while/Switch:1
Brnn_classification/cell_0/bidirectional_rnn/bw/bw/while/Switch_1:0
Brnn_classification/cell_0/bidirectional_rnn/bw/bw/while/Switch_1:1
Brnn_classification/cell_0/bidirectional_rnn/bw/bw/while/Switch_2:0
Brnn_classification/cell_0/bidirectional_rnn/bw/bw/while/Switch_2:1
Brnn_classification/cell_0/bidirectional_rnn/bw/bw/while/Switch_3:0
Brnn_classification/cell_0/bidirectional_rnn/bw/bw/while/Switch_3:1
Brnn_classification/cell_0/bidirectional_rnn/bw/bw/while/Switch_4:0
Brnn_classification/cell_0/bidirectional_rnn/bw/bw/while/Switch_4:1
Qrnn_classification/cell_0/bidirectional_rnn/bw/bw/while/TensorArrayReadV3/Enter:0
Srnn_classification/cell_0/bidirectional_rnn/bw/bw/while/TensorArrayReadV3/Enter_1:0
Krnn_classification/cell_0/bidirectional_rnn/bw/bw/while/TensorArrayReadV3:0
crnn_classification/cell_0/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3/Enter:0
]rnn_classification/cell_0/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3:0
?rnn_classification/cell_0/bidirectional_rnn/bw/bw/while/add/y:0
=rnn_classification/cell_0/bidirectional_rnn/bw/bw/while/add:0
Arnn_classification/cell_0/bidirectional_rnn/bw/bw/while/add_1/y:0
?rnn_classification/cell_0/bidirectional_rnn/bw/bw/while/add_1:0
Mrnn_classification/cell_0/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Add:0
Ornn_classification/cell_0/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Add_1:0
Wrnn_classification/cell_0/bidirectional_rnn/bw/bw/while/basic_lstm_cell/BiasAdd/Enter:0
Qrnn_classification/cell_0/bidirectional_rnn/bw/bw/while/basic_lstm_cell/BiasAdd:0
Ornn_classification/cell_0/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Const:0
Qrnn_classification/cell_0/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Const_1:0
Qrnn_classification/cell_0/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Const_2:0
Vrnn_classification/cell_0/bidirectional_rnn/bw/bw/while/basic_lstm_cell/MatMul/Enter:0
Prnn_classification/cell_0/bidirectional_rnn/bw/bw/while/basic_lstm_cell/MatMul:0
Mrnn_classification/cell_0/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul:0
Ornn_classification/cell_0/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_1:0
Ornn_classification/cell_0/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_2:0
Qrnn_classification/cell_0/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Sigmoid:0
Srnn_classification/cell_0/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Sigmoid_1:0
Srnn_classification/cell_0/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Sigmoid_2:0
Nrnn_classification/cell_0/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Tanh:0
Prnn_classification/cell_0/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Tanh_1:0
Urnn_classification/cell_0/bidirectional_rnn/bw/bw/while/basic_lstm_cell/concat/axis:0
Prnn_classification/cell_0/bidirectional_rnn/bw/bw/while/basic_lstm_cell/concat:0
Ornn_classification/cell_0/bidirectional_rnn/bw/bw/while/basic_lstm_cell/split:0
Ornn_classification/cell_0/bidirectional_rnn/bw/bw/while/basic_lstm_cell/split:1
Ornn_classification/cell_0/bidirectional_rnn/bw/bw/while/basic_lstm_cell/split:2
Ornn_classification/cell_0/bidirectional_rnn/bw/bw/while/basic_lstm_cell/split:3
9rnn_classification/cell_0/bidirectional_rnn/bw/bw/zeros:0Ľ
Jrnn_classification/cell_0/bidirectional_rnn/bw/basic_lstm_cell/bias/read:0Wrnn_classification/cell_0/bidirectional_rnn/bw/bw/while/basic_lstm_cell/BiasAdd/Enter:0
9rnn_classification/cell_0/bidirectional_rnn/bw/bw/zeros:0Frnn_classification/cell_0/bidirectional_rnn/bw/bw/while/Select/Enter:0Ś
?rnn_classification/cell_0/bidirectional_rnn/bw/bw/TensorArray:0crnn_classification/cell_0/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3/Enter:0
;rnn_classification/cell_0/bidirectional_rnn/bw/bw/Minimum:0Frnn_classification/cell_0/bidirectional_rnn/bw/bw/while/Less_1/Enter:0
Arnn_classification/cell_0/bidirectional_rnn/bw/bw/TensorArray_1:0Qrnn_classification/cell_0/bidirectional_rnn/bw/bw/while/TensorArrayReadV3/Enter:0Ĺ
nrnn_classification/cell_0/bidirectional_rnn/bw/bw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3:0Srnn_classification/cell_0/bidirectional_rnn/bw/bw/while/TensorArrayReadV3/Enter_1:0
Arnn_classification/cell_0/bidirectional_rnn/bw/bw/strided_slice:0Drnn_classification/cell_0/bidirectional_rnn/bw/bw/while/Less/Enter:0
?rnn_classification/cell_0/bidirectional_rnn/bw/bw/CheckSeqLen:0Lrnn_classification/cell_0/bidirectional_rnn/bw/bw/while/GreaterEqual/Enter:0Ś
Lrnn_classification/cell_0/bidirectional_rnn/bw/basic_lstm_cell/kernel/read:0Vrnn_classification/cell_0/bidirectional_rnn/bw/bw/while/basic_lstm_cell/MatMul/Enter:0R?rnn_classification/cell_0/bidirectional_rnn/bw/bw/while/Enter:0RArnn_classification/cell_0/bidirectional_rnn/bw/bw/while/Enter_1:0RArnn_classification/cell_0/bidirectional_rnn/bw/bw/while/Enter_2:0RArnn_classification/cell_0/bidirectional_rnn/bw/bw/while/Enter_3:0RArnn_classification/cell_0/bidirectional_rnn/bw/bw/while/Enter_4:0ZArnn_classification/cell_0/bidirectional_rnn/bw/bw/strided_slice:0
H
Ernn_classification/cell_1/bidirectional_rnn/fw/fw/while/while_context *Brnn_classification/cell_1/bidirectional_rnn/fw/fw/while/LoopCond:02?rnn_classification/cell_1/bidirectional_rnn/fw/fw/while/Merge:0:Brnn_classification/cell_1/bidirectional_rnn/fw/fw/while/Identity:0B>rnn_classification/cell_1/bidirectional_rnn/fw/fw/while/Exit:0B@rnn_classification/cell_1/bidirectional_rnn/fw/fw/while/Exit_1:0B@rnn_classification/cell_1/bidirectional_rnn/fw/fw/while/Exit_2:0B@rnn_classification/cell_1/bidirectional_rnn/fw/fw/while/Exit_3:0B@rnn_classification/cell_1/bidirectional_rnn/fw/fw/while/Exit_4:0J@
Jrnn_classification/cell_1/bidirectional_rnn/fw/basic_lstm_cell/bias/read:0
Lrnn_classification/cell_1/bidirectional_rnn/fw/basic_lstm_cell/kernel/read:0
?rnn_classification/cell_1/bidirectional_rnn/fw/fw/CheckSeqLen:0
;rnn_classification/cell_1/bidirectional_rnn/fw/fw/Minimum:0
?rnn_classification/cell_1/bidirectional_rnn/fw/fw/TensorArray:0
nrnn_classification/cell_1/bidirectional_rnn/fw/fw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3:0
Arnn_classification/cell_1/bidirectional_rnn/fw/fw/TensorArray_1:0
Arnn_classification/cell_1/bidirectional_rnn/fw/fw/strided_slice:0
?rnn_classification/cell_1/bidirectional_rnn/fw/fw/while/Enter:0
Arnn_classification/cell_1/bidirectional_rnn/fw/fw/while/Enter_1:0
Arnn_classification/cell_1/bidirectional_rnn/fw/fw/while/Enter_2:0
Arnn_classification/cell_1/bidirectional_rnn/fw/fw/while/Enter_3:0
Arnn_classification/cell_1/bidirectional_rnn/fw/fw/while/Enter_4:0
>rnn_classification/cell_1/bidirectional_rnn/fw/fw/while/Exit:0
@rnn_classification/cell_1/bidirectional_rnn/fw/fw/while/Exit_1:0
@rnn_classification/cell_1/bidirectional_rnn/fw/fw/while/Exit_2:0
@rnn_classification/cell_1/bidirectional_rnn/fw/fw/while/Exit_3:0
@rnn_classification/cell_1/bidirectional_rnn/fw/fw/while/Exit_4:0
Lrnn_classification/cell_1/bidirectional_rnn/fw/fw/while/GreaterEqual/Enter:0
Frnn_classification/cell_1/bidirectional_rnn/fw/fw/while/GreaterEqual:0
Brnn_classification/cell_1/bidirectional_rnn/fw/fw/while/Identity:0
Drnn_classification/cell_1/bidirectional_rnn/fw/fw/while/Identity_1:0
Drnn_classification/cell_1/bidirectional_rnn/fw/fw/while/Identity_2:0
Drnn_classification/cell_1/bidirectional_rnn/fw/fw/while/Identity_3:0
Drnn_classification/cell_1/bidirectional_rnn/fw/fw/while/Identity_4:0
Drnn_classification/cell_1/bidirectional_rnn/fw/fw/while/Less/Enter:0
>rnn_classification/cell_1/bidirectional_rnn/fw/fw/while/Less:0
Frnn_classification/cell_1/bidirectional_rnn/fw/fw/while/Less_1/Enter:0
@rnn_classification/cell_1/bidirectional_rnn/fw/fw/while/Less_1:0
Drnn_classification/cell_1/bidirectional_rnn/fw/fw/while/LogicalAnd:0
Brnn_classification/cell_1/bidirectional_rnn/fw/fw/while/LoopCond:0
?rnn_classification/cell_1/bidirectional_rnn/fw/fw/while/Merge:0
?rnn_classification/cell_1/bidirectional_rnn/fw/fw/while/Merge:1
Arnn_classification/cell_1/bidirectional_rnn/fw/fw/while/Merge_1:0
Arnn_classification/cell_1/bidirectional_rnn/fw/fw/while/Merge_1:1
Arnn_classification/cell_1/bidirectional_rnn/fw/fw/while/Merge_2:0
Arnn_classification/cell_1/bidirectional_rnn/fw/fw/while/Merge_2:1
Arnn_classification/cell_1/bidirectional_rnn/fw/fw/while/Merge_3:0
Arnn_classification/cell_1/bidirectional_rnn/fw/fw/while/Merge_3:1
Arnn_classification/cell_1/bidirectional_rnn/fw/fw/while/Merge_4:0
Arnn_classification/cell_1/bidirectional_rnn/fw/fw/while/Merge_4:1
Grnn_classification/cell_1/bidirectional_rnn/fw/fw/while/NextIteration:0
Irnn_classification/cell_1/bidirectional_rnn/fw/fw/while/NextIteration_1:0
Irnn_classification/cell_1/bidirectional_rnn/fw/fw/while/NextIteration_2:0
Irnn_classification/cell_1/bidirectional_rnn/fw/fw/while/NextIteration_3:0
Irnn_classification/cell_1/bidirectional_rnn/fw/fw/while/NextIteration_4:0
Frnn_classification/cell_1/bidirectional_rnn/fw/fw/while/Select/Enter:0
@rnn_classification/cell_1/bidirectional_rnn/fw/fw/while/Select:0
Brnn_classification/cell_1/bidirectional_rnn/fw/fw/while/Select_1:0
Brnn_classification/cell_1/bidirectional_rnn/fw/fw/while/Select_2:0
@rnn_classification/cell_1/bidirectional_rnn/fw/fw/while/Switch:0
@rnn_classification/cell_1/bidirectional_rnn/fw/fw/while/Switch:1
Brnn_classification/cell_1/bidirectional_rnn/fw/fw/while/Switch_1:0
Brnn_classification/cell_1/bidirectional_rnn/fw/fw/while/Switch_1:1
Brnn_classification/cell_1/bidirectional_rnn/fw/fw/while/Switch_2:0
Brnn_classification/cell_1/bidirectional_rnn/fw/fw/while/Switch_2:1
Brnn_classification/cell_1/bidirectional_rnn/fw/fw/while/Switch_3:0
Brnn_classification/cell_1/bidirectional_rnn/fw/fw/while/Switch_3:1
Brnn_classification/cell_1/bidirectional_rnn/fw/fw/while/Switch_4:0
Brnn_classification/cell_1/bidirectional_rnn/fw/fw/while/Switch_4:1
Qrnn_classification/cell_1/bidirectional_rnn/fw/fw/while/TensorArrayReadV3/Enter:0
Srnn_classification/cell_1/bidirectional_rnn/fw/fw/while/TensorArrayReadV3/Enter_1:0
Krnn_classification/cell_1/bidirectional_rnn/fw/fw/while/TensorArrayReadV3:0
crnn_classification/cell_1/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3/Enter:0
]rnn_classification/cell_1/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3:0
?rnn_classification/cell_1/bidirectional_rnn/fw/fw/while/add/y:0
=rnn_classification/cell_1/bidirectional_rnn/fw/fw/while/add:0
Arnn_classification/cell_1/bidirectional_rnn/fw/fw/while/add_1/y:0
?rnn_classification/cell_1/bidirectional_rnn/fw/fw/while/add_1:0
Mrnn_classification/cell_1/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Add:0
Ornn_classification/cell_1/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Add_1:0
Wrnn_classification/cell_1/bidirectional_rnn/fw/fw/while/basic_lstm_cell/BiasAdd/Enter:0
Qrnn_classification/cell_1/bidirectional_rnn/fw/fw/while/basic_lstm_cell/BiasAdd:0
Ornn_classification/cell_1/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Const:0
Qrnn_classification/cell_1/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Const_1:0
Qrnn_classification/cell_1/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Const_2:0
Vrnn_classification/cell_1/bidirectional_rnn/fw/fw/while/basic_lstm_cell/MatMul/Enter:0
Prnn_classification/cell_1/bidirectional_rnn/fw/fw/while/basic_lstm_cell/MatMul:0
Mrnn_classification/cell_1/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul:0
Ornn_classification/cell_1/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_1:0
Ornn_classification/cell_1/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_2:0
Qrnn_classification/cell_1/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Sigmoid:0
Srnn_classification/cell_1/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Sigmoid_1:0
Srnn_classification/cell_1/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Sigmoid_2:0
Nrnn_classification/cell_1/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Tanh:0
Prnn_classification/cell_1/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Tanh_1:0
Urnn_classification/cell_1/bidirectional_rnn/fw/fw/while/basic_lstm_cell/concat/axis:0
Prnn_classification/cell_1/bidirectional_rnn/fw/fw/while/basic_lstm_cell/concat:0
Ornn_classification/cell_1/bidirectional_rnn/fw/fw/while/basic_lstm_cell/split:0
Ornn_classification/cell_1/bidirectional_rnn/fw/fw/while/basic_lstm_cell/split:1
Ornn_classification/cell_1/bidirectional_rnn/fw/fw/while/basic_lstm_cell/split:2
Ornn_classification/cell_1/bidirectional_rnn/fw/fw/while/basic_lstm_cell/split:3
9rnn_classification/cell_1/bidirectional_rnn/fw/fw/zeros:0
9rnn_classification/cell_1/bidirectional_rnn/fw/fw/zeros:0Frnn_classification/cell_1/bidirectional_rnn/fw/fw/while/Select/Enter:0
?rnn_classification/cell_1/bidirectional_rnn/fw/fw/CheckSeqLen:0Lrnn_classification/cell_1/bidirectional_rnn/fw/fw/while/GreaterEqual/Enter:0Ś
Lrnn_classification/cell_1/bidirectional_rnn/fw/basic_lstm_cell/kernel/read:0Vrnn_classification/cell_1/bidirectional_rnn/fw/fw/while/basic_lstm_cell/MatMul/Enter:0
Arnn_classification/cell_1/bidirectional_rnn/fw/fw/TensorArray_1:0Qrnn_classification/cell_1/bidirectional_rnn/fw/fw/while/TensorArrayReadV3/Enter:0Ś
?rnn_classification/cell_1/bidirectional_rnn/fw/fw/TensorArray:0crnn_classification/cell_1/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3/Enter:0
Arnn_classification/cell_1/bidirectional_rnn/fw/fw/strided_slice:0Drnn_classification/cell_1/bidirectional_rnn/fw/fw/while/Less/Enter:0Ĺ
nrnn_classification/cell_1/bidirectional_rnn/fw/fw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3:0Srnn_classification/cell_1/bidirectional_rnn/fw/fw/while/TensorArrayReadV3/Enter_1:0
;rnn_classification/cell_1/bidirectional_rnn/fw/fw/Minimum:0Frnn_classification/cell_1/bidirectional_rnn/fw/fw/while/Less_1/Enter:0Ľ
Jrnn_classification/cell_1/bidirectional_rnn/fw/basic_lstm_cell/bias/read:0Wrnn_classification/cell_1/bidirectional_rnn/fw/fw/while/basic_lstm_cell/BiasAdd/Enter:0R?rnn_classification/cell_1/bidirectional_rnn/fw/fw/while/Enter:0RArnn_classification/cell_1/bidirectional_rnn/fw/fw/while/Enter_1:0RArnn_classification/cell_1/bidirectional_rnn/fw/fw/while/Enter_2:0RArnn_classification/cell_1/bidirectional_rnn/fw/fw/while/Enter_3:0RArnn_classification/cell_1/bidirectional_rnn/fw/fw/while/Enter_4:0ZArnn_classification/cell_1/bidirectional_rnn/fw/fw/strided_slice:0
H
Ernn_classification/cell_1/bidirectional_rnn/bw/bw/while/while_context *Brnn_classification/cell_1/bidirectional_rnn/bw/bw/while/LoopCond:02?rnn_classification/cell_1/bidirectional_rnn/bw/bw/while/Merge:0:Brnn_classification/cell_1/bidirectional_rnn/bw/bw/while/Identity:0B>rnn_classification/cell_1/bidirectional_rnn/bw/bw/while/Exit:0B@rnn_classification/cell_1/bidirectional_rnn/bw/bw/while/Exit_1:0B@rnn_classification/cell_1/bidirectional_rnn/bw/bw/while/Exit_2:0B@rnn_classification/cell_1/bidirectional_rnn/bw/bw/while/Exit_3:0B@rnn_classification/cell_1/bidirectional_rnn/bw/bw/while/Exit_4:0J@
Jrnn_classification/cell_1/bidirectional_rnn/bw/basic_lstm_cell/bias/read:0
Lrnn_classification/cell_1/bidirectional_rnn/bw/basic_lstm_cell/kernel/read:0
?rnn_classification/cell_1/bidirectional_rnn/bw/bw/CheckSeqLen:0
;rnn_classification/cell_1/bidirectional_rnn/bw/bw/Minimum:0
?rnn_classification/cell_1/bidirectional_rnn/bw/bw/TensorArray:0
nrnn_classification/cell_1/bidirectional_rnn/bw/bw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3:0
Arnn_classification/cell_1/bidirectional_rnn/bw/bw/TensorArray_1:0
Arnn_classification/cell_1/bidirectional_rnn/bw/bw/strided_slice:0
?rnn_classification/cell_1/bidirectional_rnn/bw/bw/while/Enter:0
Arnn_classification/cell_1/bidirectional_rnn/bw/bw/while/Enter_1:0
Arnn_classification/cell_1/bidirectional_rnn/bw/bw/while/Enter_2:0
Arnn_classification/cell_1/bidirectional_rnn/bw/bw/while/Enter_3:0
Arnn_classification/cell_1/bidirectional_rnn/bw/bw/while/Enter_4:0
>rnn_classification/cell_1/bidirectional_rnn/bw/bw/while/Exit:0
@rnn_classification/cell_1/bidirectional_rnn/bw/bw/while/Exit_1:0
@rnn_classification/cell_1/bidirectional_rnn/bw/bw/while/Exit_2:0
@rnn_classification/cell_1/bidirectional_rnn/bw/bw/while/Exit_3:0
@rnn_classification/cell_1/bidirectional_rnn/bw/bw/while/Exit_4:0
Lrnn_classification/cell_1/bidirectional_rnn/bw/bw/while/GreaterEqual/Enter:0
Frnn_classification/cell_1/bidirectional_rnn/bw/bw/while/GreaterEqual:0
Brnn_classification/cell_1/bidirectional_rnn/bw/bw/while/Identity:0
Drnn_classification/cell_1/bidirectional_rnn/bw/bw/while/Identity_1:0
Drnn_classification/cell_1/bidirectional_rnn/bw/bw/while/Identity_2:0
Drnn_classification/cell_1/bidirectional_rnn/bw/bw/while/Identity_3:0
Drnn_classification/cell_1/bidirectional_rnn/bw/bw/while/Identity_4:0
Drnn_classification/cell_1/bidirectional_rnn/bw/bw/while/Less/Enter:0
>rnn_classification/cell_1/bidirectional_rnn/bw/bw/while/Less:0
Frnn_classification/cell_1/bidirectional_rnn/bw/bw/while/Less_1/Enter:0
@rnn_classification/cell_1/bidirectional_rnn/bw/bw/while/Less_1:0
Drnn_classification/cell_1/bidirectional_rnn/bw/bw/while/LogicalAnd:0
Brnn_classification/cell_1/bidirectional_rnn/bw/bw/while/LoopCond:0
?rnn_classification/cell_1/bidirectional_rnn/bw/bw/while/Merge:0
?rnn_classification/cell_1/bidirectional_rnn/bw/bw/while/Merge:1
Arnn_classification/cell_1/bidirectional_rnn/bw/bw/while/Merge_1:0
Arnn_classification/cell_1/bidirectional_rnn/bw/bw/while/Merge_1:1
Arnn_classification/cell_1/bidirectional_rnn/bw/bw/while/Merge_2:0
Arnn_classification/cell_1/bidirectional_rnn/bw/bw/while/Merge_2:1
Arnn_classification/cell_1/bidirectional_rnn/bw/bw/while/Merge_3:0
Arnn_classification/cell_1/bidirectional_rnn/bw/bw/while/Merge_3:1
Arnn_classification/cell_1/bidirectional_rnn/bw/bw/while/Merge_4:0
Arnn_classification/cell_1/bidirectional_rnn/bw/bw/while/Merge_4:1
Grnn_classification/cell_1/bidirectional_rnn/bw/bw/while/NextIteration:0
Irnn_classification/cell_1/bidirectional_rnn/bw/bw/while/NextIteration_1:0
Irnn_classification/cell_1/bidirectional_rnn/bw/bw/while/NextIteration_2:0
Irnn_classification/cell_1/bidirectional_rnn/bw/bw/while/NextIteration_3:0
Irnn_classification/cell_1/bidirectional_rnn/bw/bw/while/NextIteration_4:0
Frnn_classification/cell_1/bidirectional_rnn/bw/bw/while/Select/Enter:0
@rnn_classification/cell_1/bidirectional_rnn/bw/bw/while/Select:0
Brnn_classification/cell_1/bidirectional_rnn/bw/bw/while/Select_1:0
Brnn_classification/cell_1/bidirectional_rnn/bw/bw/while/Select_2:0
@rnn_classification/cell_1/bidirectional_rnn/bw/bw/while/Switch:0
@rnn_classification/cell_1/bidirectional_rnn/bw/bw/while/Switch:1
Brnn_classification/cell_1/bidirectional_rnn/bw/bw/while/Switch_1:0
Brnn_classification/cell_1/bidirectional_rnn/bw/bw/while/Switch_1:1
Brnn_classification/cell_1/bidirectional_rnn/bw/bw/while/Switch_2:0
Brnn_classification/cell_1/bidirectional_rnn/bw/bw/while/Switch_2:1
Brnn_classification/cell_1/bidirectional_rnn/bw/bw/while/Switch_3:0
Brnn_classification/cell_1/bidirectional_rnn/bw/bw/while/Switch_3:1
Brnn_classification/cell_1/bidirectional_rnn/bw/bw/while/Switch_4:0
Brnn_classification/cell_1/bidirectional_rnn/bw/bw/while/Switch_4:1
Qrnn_classification/cell_1/bidirectional_rnn/bw/bw/while/TensorArrayReadV3/Enter:0
Srnn_classification/cell_1/bidirectional_rnn/bw/bw/while/TensorArrayReadV3/Enter_1:0
Krnn_classification/cell_1/bidirectional_rnn/bw/bw/while/TensorArrayReadV3:0
crnn_classification/cell_1/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3/Enter:0
]rnn_classification/cell_1/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3:0
?rnn_classification/cell_1/bidirectional_rnn/bw/bw/while/add/y:0
=rnn_classification/cell_1/bidirectional_rnn/bw/bw/while/add:0
Arnn_classification/cell_1/bidirectional_rnn/bw/bw/while/add_1/y:0
?rnn_classification/cell_1/bidirectional_rnn/bw/bw/while/add_1:0
Mrnn_classification/cell_1/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Add:0
Ornn_classification/cell_1/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Add_1:0
Wrnn_classification/cell_1/bidirectional_rnn/bw/bw/while/basic_lstm_cell/BiasAdd/Enter:0
Qrnn_classification/cell_1/bidirectional_rnn/bw/bw/while/basic_lstm_cell/BiasAdd:0
Ornn_classification/cell_1/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Const:0
Qrnn_classification/cell_1/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Const_1:0
Qrnn_classification/cell_1/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Const_2:0
Vrnn_classification/cell_1/bidirectional_rnn/bw/bw/while/basic_lstm_cell/MatMul/Enter:0
Prnn_classification/cell_1/bidirectional_rnn/bw/bw/while/basic_lstm_cell/MatMul:0
Mrnn_classification/cell_1/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul:0
Ornn_classification/cell_1/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_1:0
Ornn_classification/cell_1/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_2:0
Qrnn_classification/cell_1/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Sigmoid:0
Srnn_classification/cell_1/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Sigmoid_1:0
Srnn_classification/cell_1/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Sigmoid_2:0
Nrnn_classification/cell_1/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Tanh:0
Prnn_classification/cell_1/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Tanh_1:0
Urnn_classification/cell_1/bidirectional_rnn/bw/bw/while/basic_lstm_cell/concat/axis:0
Prnn_classification/cell_1/bidirectional_rnn/bw/bw/while/basic_lstm_cell/concat:0
Ornn_classification/cell_1/bidirectional_rnn/bw/bw/while/basic_lstm_cell/split:0
Ornn_classification/cell_1/bidirectional_rnn/bw/bw/while/basic_lstm_cell/split:1
Ornn_classification/cell_1/bidirectional_rnn/bw/bw/while/basic_lstm_cell/split:2
Ornn_classification/cell_1/bidirectional_rnn/bw/bw/while/basic_lstm_cell/split:3
9rnn_classification/cell_1/bidirectional_rnn/bw/bw/zeros:0Ś
?rnn_classification/cell_1/bidirectional_rnn/bw/bw/TensorArray:0crnn_classification/cell_1/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3/Enter:0
Arnn_classification/cell_1/bidirectional_rnn/bw/bw/strided_slice:0Drnn_classification/cell_1/bidirectional_rnn/bw/bw/while/Less/Enter:0Ĺ
nrnn_classification/cell_1/bidirectional_rnn/bw/bw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3:0Srnn_classification/cell_1/bidirectional_rnn/bw/bw/while/TensorArrayReadV3/Enter_1:0
;rnn_classification/cell_1/bidirectional_rnn/bw/bw/Minimum:0Frnn_classification/cell_1/bidirectional_rnn/bw/bw/while/Less_1/Enter:0
9rnn_classification/cell_1/bidirectional_rnn/bw/bw/zeros:0Frnn_classification/cell_1/bidirectional_rnn/bw/bw/while/Select/Enter:0Ľ
Jrnn_classification/cell_1/bidirectional_rnn/bw/basic_lstm_cell/bias/read:0Wrnn_classification/cell_1/bidirectional_rnn/bw/bw/while/basic_lstm_cell/BiasAdd/Enter:0
?rnn_classification/cell_1/bidirectional_rnn/bw/bw/CheckSeqLen:0Lrnn_classification/cell_1/bidirectional_rnn/bw/bw/while/GreaterEqual/Enter:0
Arnn_classification/cell_1/bidirectional_rnn/bw/bw/TensorArray_1:0Qrnn_classification/cell_1/bidirectional_rnn/bw/bw/while/TensorArrayReadV3/Enter:0Ś
Lrnn_classification/cell_1/bidirectional_rnn/bw/basic_lstm_cell/kernel/read:0Vrnn_classification/cell_1/bidirectional_rnn/bw/bw/while/basic_lstm_cell/MatMul/Enter:0R?rnn_classification/cell_1/bidirectional_rnn/bw/bw/while/Enter:0RArnn_classification/cell_1/bidirectional_rnn/bw/bw/while/Enter_1:0RArnn_classification/cell_1/bidirectional_rnn/bw/bw/while/Enter_2:0RArnn_classification/cell_1/bidirectional_rnn/bw/bw/while/Enter_3:0RArnn_classification/cell_1/bidirectional_rnn/bw/bw/while/Enter_4:0ZArnn_classification/cell_1/bidirectional_rnn/bw/bw/strided_slice:0
H
Ernn_classification/cell_2/bidirectional_rnn/fw/fw/while/while_context *Brnn_classification/cell_2/bidirectional_rnn/fw/fw/while/LoopCond:02?rnn_classification/cell_2/bidirectional_rnn/fw/fw/while/Merge:0:Brnn_classification/cell_2/bidirectional_rnn/fw/fw/while/Identity:0B>rnn_classification/cell_2/bidirectional_rnn/fw/fw/while/Exit:0B@rnn_classification/cell_2/bidirectional_rnn/fw/fw/while/Exit_1:0B@rnn_classification/cell_2/bidirectional_rnn/fw/fw/while/Exit_2:0B@rnn_classification/cell_2/bidirectional_rnn/fw/fw/while/Exit_3:0B@rnn_classification/cell_2/bidirectional_rnn/fw/fw/while/Exit_4:0J@
Jrnn_classification/cell_2/bidirectional_rnn/fw/basic_lstm_cell/bias/read:0
Lrnn_classification/cell_2/bidirectional_rnn/fw/basic_lstm_cell/kernel/read:0
?rnn_classification/cell_2/bidirectional_rnn/fw/fw/CheckSeqLen:0
;rnn_classification/cell_2/bidirectional_rnn/fw/fw/Minimum:0
?rnn_classification/cell_2/bidirectional_rnn/fw/fw/TensorArray:0
nrnn_classification/cell_2/bidirectional_rnn/fw/fw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3:0
Arnn_classification/cell_2/bidirectional_rnn/fw/fw/TensorArray_1:0
Arnn_classification/cell_2/bidirectional_rnn/fw/fw/strided_slice:0
?rnn_classification/cell_2/bidirectional_rnn/fw/fw/while/Enter:0
Arnn_classification/cell_2/bidirectional_rnn/fw/fw/while/Enter_1:0
Arnn_classification/cell_2/bidirectional_rnn/fw/fw/while/Enter_2:0
Arnn_classification/cell_2/bidirectional_rnn/fw/fw/while/Enter_3:0
Arnn_classification/cell_2/bidirectional_rnn/fw/fw/while/Enter_4:0
>rnn_classification/cell_2/bidirectional_rnn/fw/fw/while/Exit:0
@rnn_classification/cell_2/bidirectional_rnn/fw/fw/while/Exit_1:0
@rnn_classification/cell_2/bidirectional_rnn/fw/fw/while/Exit_2:0
@rnn_classification/cell_2/bidirectional_rnn/fw/fw/while/Exit_3:0
@rnn_classification/cell_2/bidirectional_rnn/fw/fw/while/Exit_4:0
Lrnn_classification/cell_2/bidirectional_rnn/fw/fw/while/GreaterEqual/Enter:0
Frnn_classification/cell_2/bidirectional_rnn/fw/fw/while/GreaterEqual:0
Brnn_classification/cell_2/bidirectional_rnn/fw/fw/while/Identity:0
Drnn_classification/cell_2/bidirectional_rnn/fw/fw/while/Identity_1:0
Drnn_classification/cell_2/bidirectional_rnn/fw/fw/while/Identity_2:0
Drnn_classification/cell_2/bidirectional_rnn/fw/fw/while/Identity_3:0
Drnn_classification/cell_2/bidirectional_rnn/fw/fw/while/Identity_4:0
Drnn_classification/cell_2/bidirectional_rnn/fw/fw/while/Less/Enter:0
>rnn_classification/cell_2/bidirectional_rnn/fw/fw/while/Less:0
Frnn_classification/cell_2/bidirectional_rnn/fw/fw/while/Less_1/Enter:0
@rnn_classification/cell_2/bidirectional_rnn/fw/fw/while/Less_1:0
Drnn_classification/cell_2/bidirectional_rnn/fw/fw/while/LogicalAnd:0
Brnn_classification/cell_2/bidirectional_rnn/fw/fw/while/LoopCond:0
?rnn_classification/cell_2/bidirectional_rnn/fw/fw/while/Merge:0
?rnn_classification/cell_2/bidirectional_rnn/fw/fw/while/Merge:1
Arnn_classification/cell_2/bidirectional_rnn/fw/fw/while/Merge_1:0
Arnn_classification/cell_2/bidirectional_rnn/fw/fw/while/Merge_1:1
Arnn_classification/cell_2/bidirectional_rnn/fw/fw/while/Merge_2:0
Arnn_classification/cell_2/bidirectional_rnn/fw/fw/while/Merge_2:1
Arnn_classification/cell_2/bidirectional_rnn/fw/fw/while/Merge_3:0
Arnn_classification/cell_2/bidirectional_rnn/fw/fw/while/Merge_3:1
Arnn_classification/cell_2/bidirectional_rnn/fw/fw/while/Merge_4:0
Arnn_classification/cell_2/bidirectional_rnn/fw/fw/while/Merge_4:1
Grnn_classification/cell_2/bidirectional_rnn/fw/fw/while/NextIteration:0
Irnn_classification/cell_2/bidirectional_rnn/fw/fw/while/NextIteration_1:0
Irnn_classification/cell_2/bidirectional_rnn/fw/fw/while/NextIteration_2:0
Irnn_classification/cell_2/bidirectional_rnn/fw/fw/while/NextIteration_3:0
Irnn_classification/cell_2/bidirectional_rnn/fw/fw/while/NextIteration_4:0
Frnn_classification/cell_2/bidirectional_rnn/fw/fw/while/Select/Enter:0
@rnn_classification/cell_2/bidirectional_rnn/fw/fw/while/Select:0
Brnn_classification/cell_2/bidirectional_rnn/fw/fw/while/Select_1:0
Brnn_classification/cell_2/bidirectional_rnn/fw/fw/while/Select_2:0
@rnn_classification/cell_2/bidirectional_rnn/fw/fw/while/Switch:0
@rnn_classification/cell_2/bidirectional_rnn/fw/fw/while/Switch:1
Brnn_classification/cell_2/bidirectional_rnn/fw/fw/while/Switch_1:0
Brnn_classification/cell_2/bidirectional_rnn/fw/fw/while/Switch_1:1
Brnn_classification/cell_2/bidirectional_rnn/fw/fw/while/Switch_2:0
Brnn_classification/cell_2/bidirectional_rnn/fw/fw/while/Switch_2:1
Brnn_classification/cell_2/bidirectional_rnn/fw/fw/while/Switch_3:0
Brnn_classification/cell_2/bidirectional_rnn/fw/fw/while/Switch_3:1
Brnn_classification/cell_2/bidirectional_rnn/fw/fw/while/Switch_4:0
Brnn_classification/cell_2/bidirectional_rnn/fw/fw/while/Switch_4:1
Qrnn_classification/cell_2/bidirectional_rnn/fw/fw/while/TensorArrayReadV3/Enter:0
Srnn_classification/cell_2/bidirectional_rnn/fw/fw/while/TensorArrayReadV3/Enter_1:0
Krnn_classification/cell_2/bidirectional_rnn/fw/fw/while/TensorArrayReadV3:0
crnn_classification/cell_2/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3/Enter:0
]rnn_classification/cell_2/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3:0
?rnn_classification/cell_2/bidirectional_rnn/fw/fw/while/add/y:0
=rnn_classification/cell_2/bidirectional_rnn/fw/fw/while/add:0
Arnn_classification/cell_2/bidirectional_rnn/fw/fw/while/add_1/y:0
?rnn_classification/cell_2/bidirectional_rnn/fw/fw/while/add_1:0
Mrnn_classification/cell_2/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Add:0
Ornn_classification/cell_2/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Add_1:0
Wrnn_classification/cell_2/bidirectional_rnn/fw/fw/while/basic_lstm_cell/BiasAdd/Enter:0
Qrnn_classification/cell_2/bidirectional_rnn/fw/fw/while/basic_lstm_cell/BiasAdd:0
Ornn_classification/cell_2/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Const:0
Qrnn_classification/cell_2/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Const_1:0
Qrnn_classification/cell_2/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Const_2:0
Vrnn_classification/cell_2/bidirectional_rnn/fw/fw/while/basic_lstm_cell/MatMul/Enter:0
Prnn_classification/cell_2/bidirectional_rnn/fw/fw/while/basic_lstm_cell/MatMul:0
Mrnn_classification/cell_2/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul:0
Ornn_classification/cell_2/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_1:0
Ornn_classification/cell_2/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Mul_2:0
Qrnn_classification/cell_2/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Sigmoid:0
Srnn_classification/cell_2/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Sigmoid_1:0
Srnn_classification/cell_2/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Sigmoid_2:0
Nrnn_classification/cell_2/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Tanh:0
Prnn_classification/cell_2/bidirectional_rnn/fw/fw/while/basic_lstm_cell/Tanh_1:0
Urnn_classification/cell_2/bidirectional_rnn/fw/fw/while/basic_lstm_cell/concat/axis:0
Prnn_classification/cell_2/bidirectional_rnn/fw/fw/while/basic_lstm_cell/concat:0
Ornn_classification/cell_2/bidirectional_rnn/fw/fw/while/basic_lstm_cell/split:0
Ornn_classification/cell_2/bidirectional_rnn/fw/fw/while/basic_lstm_cell/split:1
Ornn_classification/cell_2/bidirectional_rnn/fw/fw/while/basic_lstm_cell/split:2
Ornn_classification/cell_2/bidirectional_rnn/fw/fw/while/basic_lstm_cell/split:3
9rnn_classification/cell_2/bidirectional_rnn/fw/fw/zeros:0
Arnn_classification/cell_2/bidirectional_rnn/fw/fw/strided_slice:0Drnn_classification/cell_2/bidirectional_rnn/fw/fw/while/Less/Enter:0
?rnn_classification/cell_2/bidirectional_rnn/fw/fw/CheckSeqLen:0Lrnn_classification/cell_2/bidirectional_rnn/fw/fw/while/GreaterEqual/Enter:0Ĺ
nrnn_classification/cell_2/bidirectional_rnn/fw/fw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3:0Srnn_classification/cell_2/bidirectional_rnn/fw/fw/while/TensorArrayReadV3/Enter_1:0
9rnn_classification/cell_2/bidirectional_rnn/fw/fw/zeros:0Frnn_classification/cell_2/bidirectional_rnn/fw/fw/while/Select/Enter:0Ś
Lrnn_classification/cell_2/bidirectional_rnn/fw/basic_lstm_cell/kernel/read:0Vrnn_classification/cell_2/bidirectional_rnn/fw/fw/while/basic_lstm_cell/MatMul/Enter:0Ľ
Jrnn_classification/cell_2/bidirectional_rnn/fw/basic_lstm_cell/bias/read:0Wrnn_classification/cell_2/bidirectional_rnn/fw/fw/while/basic_lstm_cell/BiasAdd/Enter:0Ś
?rnn_classification/cell_2/bidirectional_rnn/fw/fw/TensorArray:0crnn_classification/cell_2/bidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3/Enter:0
;rnn_classification/cell_2/bidirectional_rnn/fw/fw/Minimum:0Frnn_classification/cell_2/bidirectional_rnn/fw/fw/while/Less_1/Enter:0
Arnn_classification/cell_2/bidirectional_rnn/fw/fw/TensorArray_1:0Qrnn_classification/cell_2/bidirectional_rnn/fw/fw/while/TensorArrayReadV3/Enter:0R?rnn_classification/cell_2/bidirectional_rnn/fw/fw/while/Enter:0RArnn_classification/cell_2/bidirectional_rnn/fw/fw/while/Enter_1:0RArnn_classification/cell_2/bidirectional_rnn/fw/fw/while/Enter_2:0RArnn_classification/cell_2/bidirectional_rnn/fw/fw/while/Enter_3:0RArnn_classification/cell_2/bidirectional_rnn/fw/fw/while/Enter_4:0ZArnn_classification/cell_2/bidirectional_rnn/fw/fw/strided_slice:0
H
Ernn_classification/cell_2/bidirectional_rnn/bw/bw/while/while_context *Brnn_classification/cell_2/bidirectional_rnn/bw/bw/while/LoopCond:02?rnn_classification/cell_2/bidirectional_rnn/bw/bw/while/Merge:0:Brnn_classification/cell_2/bidirectional_rnn/bw/bw/while/Identity:0B>rnn_classification/cell_2/bidirectional_rnn/bw/bw/while/Exit:0B@rnn_classification/cell_2/bidirectional_rnn/bw/bw/while/Exit_1:0B@rnn_classification/cell_2/bidirectional_rnn/bw/bw/while/Exit_2:0B@rnn_classification/cell_2/bidirectional_rnn/bw/bw/while/Exit_3:0B@rnn_classification/cell_2/bidirectional_rnn/bw/bw/while/Exit_4:0J@
Jrnn_classification/cell_2/bidirectional_rnn/bw/basic_lstm_cell/bias/read:0
Lrnn_classification/cell_2/bidirectional_rnn/bw/basic_lstm_cell/kernel/read:0
?rnn_classification/cell_2/bidirectional_rnn/bw/bw/CheckSeqLen:0
;rnn_classification/cell_2/bidirectional_rnn/bw/bw/Minimum:0
?rnn_classification/cell_2/bidirectional_rnn/bw/bw/TensorArray:0
nrnn_classification/cell_2/bidirectional_rnn/bw/bw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3:0
Arnn_classification/cell_2/bidirectional_rnn/bw/bw/TensorArray_1:0
Arnn_classification/cell_2/bidirectional_rnn/bw/bw/strided_slice:0
?rnn_classification/cell_2/bidirectional_rnn/bw/bw/while/Enter:0
Arnn_classification/cell_2/bidirectional_rnn/bw/bw/while/Enter_1:0
Arnn_classification/cell_2/bidirectional_rnn/bw/bw/while/Enter_2:0
Arnn_classification/cell_2/bidirectional_rnn/bw/bw/while/Enter_3:0
Arnn_classification/cell_2/bidirectional_rnn/bw/bw/while/Enter_4:0
>rnn_classification/cell_2/bidirectional_rnn/bw/bw/while/Exit:0
@rnn_classification/cell_2/bidirectional_rnn/bw/bw/while/Exit_1:0
@rnn_classification/cell_2/bidirectional_rnn/bw/bw/while/Exit_2:0
@rnn_classification/cell_2/bidirectional_rnn/bw/bw/while/Exit_3:0
@rnn_classification/cell_2/bidirectional_rnn/bw/bw/while/Exit_4:0
Lrnn_classification/cell_2/bidirectional_rnn/bw/bw/while/GreaterEqual/Enter:0
Frnn_classification/cell_2/bidirectional_rnn/bw/bw/while/GreaterEqual:0
Brnn_classification/cell_2/bidirectional_rnn/bw/bw/while/Identity:0
Drnn_classification/cell_2/bidirectional_rnn/bw/bw/while/Identity_1:0
Drnn_classification/cell_2/bidirectional_rnn/bw/bw/while/Identity_2:0
Drnn_classification/cell_2/bidirectional_rnn/bw/bw/while/Identity_3:0
Drnn_classification/cell_2/bidirectional_rnn/bw/bw/while/Identity_4:0
Drnn_classification/cell_2/bidirectional_rnn/bw/bw/while/Less/Enter:0
>rnn_classification/cell_2/bidirectional_rnn/bw/bw/while/Less:0
Frnn_classification/cell_2/bidirectional_rnn/bw/bw/while/Less_1/Enter:0
@rnn_classification/cell_2/bidirectional_rnn/bw/bw/while/Less_1:0
Drnn_classification/cell_2/bidirectional_rnn/bw/bw/while/LogicalAnd:0
Brnn_classification/cell_2/bidirectional_rnn/bw/bw/while/LoopCond:0
?rnn_classification/cell_2/bidirectional_rnn/bw/bw/while/Merge:0
?rnn_classification/cell_2/bidirectional_rnn/bw/bw/while/Merge:1
Arnn_classification/cell_2/bidirectional_rnn/bw/bw/while/Merge_1:0
Arnn_classification/cell_2/bidirectional_rnn/bw/bw/while/Merge_1:1
Arnn_classification/cell_2/bidirectional_rnn/bw/bw/while/Merge_2:0
Arnn_classification/cell_2/bidirectional_rnn/bw/bw/while/Merge_2:1
Arnn_classification/cell_2/bidirectional_rnn/bw/bw/while/Merge_3:0
Arnn_classification/cell_2/bidirectional_rnn/bw/bw/while/Merge_3:1
Arnn_classification/cell_2/bidirectional_rnn/bw/bw/while/Merge_4:0
Arnn_classification/cell_2/bidirectional_rnn/bw/bw/while/Merge_4:1
Grnn_classification/cell_2/bidirectional_rnn/bw/bw/while/NextIteration:0
Irnn_classification/cell_2/bidirectional_rnn/bw/bw/while/NextIteration_1:0
Irnn_classification/cell_2/bidirectional_rnn/bw/bw/while/NextIteration_2:0
Irnn_classification/cell_2/bidirectional_rnn/bw/bw/while/NextIteration_3:0
Irnn_classification/cell_2/bidirectional_rnn/bw/bw/while/NextIteration_4:0
Frnn_classification/cell_2/bidirectional_rnn/bw/bw/while/Select/Enter:0
@rnn_classification/cell_2/bidirectional_rnn/bw/bw/while/Select:0
Brnn_classification/cell_2/bidirectional_rnn/bw/bw/while/Select_1:0
Brnn_classification/cell_2/bidirectional_rnn/bw/bw/while/Select_2:0
@rnn_classification/cell_2/bidirectional_rnn/bw/bw/while/Switch:0
@rnn_classification/cell_2/bidirectional_rnn/bw/bw/while/Switch:1
Brnn_classification/cell_2/bidirectional_rnn/bw/bw/while/Switch_1:0
Brnn_classification/cell_2/bidirectional_rnn/bw/bw/while/Switch_1:1
Brnn_classification/cell_2/bidirectional_rnn/bw/bw/while/Switch_2:0
Brnn_classification/cell_2/bidirectional_rnn/bw/bw/while/Switch_2:1
Brnn_classification/cell_2/bidirectional_rnn/bw/bw/while/Switch_3:0
Brnn_classification/cell_2/bidirectional_rnn/bw/bw/while/Switch_3:1
Brnn_classification/cell_2/bidirectional_rnn/bw/bw/while/Switch_4:0
Brnn_classification/cell_2/bidirectional_rnn/bw/bw/while/Switch_4:1
Qrnn_classification/cell_2/bidirectional_rnn/bw/bw/while/TensorArrayReadV3/Enter:0
Srnn_classification/cell_2/bidirectional_rnn/bw/bw/while/TensorArrayReadV3/Enter_1:0
Krnn_classification/cell_2/bidirectional_rnn/bw/bw/while/TensorArrayReadV3:0
crnn_classification/cell_2/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3/Enter:0
]rnn_classification/cell_2/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3:0
?rnn_classification/cell_2/bidirectional_rnn/bw/bw/while/add/y:0
=rnn_classification/cell_2/bidirectional_rnn/bw/bw/while/add:0
Arnn_classification/cell_2/bidirectional_rnn/bw/bw/while/add_1/y:0
?rnn_classification/cell_2/bidirectional_rnn/bw/bw/while/add_1:0
Mrnn_classification/cell_2/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Add:0
Ornn_classification/cell_2/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Add_1:0
Wrnn_classification/cell_2/bidirectional_rnn/bw/bw/while/basic_lstm_cell/BiasAdd/Enter:0
Qrnn_classification/cell_2/bidirectional_rnn/bw/bw/while/basic_lstm_cell/BiasAdd:0
Ornn_classification/cell_2/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Const:0
Qrnn_classification/cell_2/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Const_1:0
Qrnn_classification/cell_2/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Const_2:0
Vrnn_classification/cell_2/bidirectional_rnn/bw/bw/while/basic_lstm_cell/MatMul/Enter:0
Prnn_classification/cell_2/bidirectional_rnn/bw/bw/while/basic_lstm_cell/MatMul:0
Mrnn_classification/cell_2/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul:0
Ornn_classification/cell_2/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_1:0
Ornn_classification/cell_2/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Mul_2:0
Qrnn_classification/cell_2/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Sigmoid:0
Srnn_classification/cell_2/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Sigmoid_1:0
Srnn_classification/cell_2/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Sigmoid_2:0
Nrnn_classification/cell_2/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Tanh:0
Prnn_classification/cell_2/bidirectional_rnn/bw/bw/while/basic_lstm_cell/Tanh_1:0
Urnn_classification/cell_2/bidirectional_rnn/bw/bw/while/basic_lstm_cell/concat/axis:0
Prnn_classification/cell_2/bidirectional_rnn/bw/bw/while/basic_lstm_cell/concat:0
Ornn_classification/cell_2/bidirectional_rnn/bw/bw/while/basic_lstm_cell/split:0
Ornn_classification/cell_2/bidirectional_rnn/bw/bw/while/basic_lstm_cell/split:1
Ornn_classification/cell_2/bidirectional_rnn/bw/bw/while/basic_lstm_cell/split:2
Ornn_classification/cell_2/bidirectional_rnn/bw/bw/while/basic_lstm_cell/split:3
9rnn_classification/cell_2/bidirectional_rnn/bw/bw/zeros:0Ĺ
nrnn_classification/cell_2/bidirectional_rnn/bw/bw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3:0Srnn_classification/cell_2/bidirectional_rnn/bw/bw/while/TensorArrayReadV3/Enter_1:0
9rnn_classification/cell_2/bidirectional_rnn/bw/bw/zeros:0Frnn_classification/cell_2/bidirectional_rnn/bw/bw/while/Select/Enter:0Ś
?rnn_classification/cell_2/bidirectional_rnn/bw/bw/TensorArray:0crnn_classification/cell_2/bidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3/Enter:0Ś
Lrnn_classification/cell_2/bidirectional_rnn/bw/basic_lstm_cell/kernel/read:0Vrnn_classification/cell_2/bidirectional_rnn/bw/bw/while/basic_lstm_cell/MatMul/Enter:0
;rnn_classification/cell_2/bidirectional_rnn/bw/bw/Minimum:0Frnn_classification/cell_2/bidirectional_rnn/bw/bw/while/Less_1/Enter:0Ľ
Jrnn_classification/cell_2/bidirectional_rnn/bw/basic_lstm_cell/bias/read:0Wrnn_classification/cell_2/bidirectional_rnn/bw/bw/while/basic_lstm_cell/BiasAdd/Enter:0
Arnn_classification/cell_2/bidirectional_rnn/bw/bw/TensorArray_1:0Qrnn_classification/cell_2/bidirectional_rnn/bw/bw/while/TensorArrayReadV3/Enter:0
Arnn_classification/cell_2/bidirectional_rnn/bw/bw/strided_slice:0Drnn_classification/cell_2/bidirectional_rnn/bw/bw/while/Less/Enter:0
?rnn_classification/cell_2/bidirectional_rnn/bw/bw/CheckSeqLen:0Lrnn_classification/cell_2/bidirectional_rnn/bw/bw/while/GreaterEqual/Enter:0R?rnn_classification/cell_2/bidirectional_rnn/bw/bw/while/Enter:0RArnn_classification/cell_2/bidirectional_rnn/bw/bw/while/Enter_1:0RArnn_classification/cell_2/bidirectional_rnn/bw/bw/while/Enter_2:0RArnn_classification/cell_2/bidirectional_rnn/bw/bw/while/Enter_3:0RArnn_classification/cell_2/bidirectional_rnn/bw/bw/while/Enter_4:0ZArnn_classification/cell_2/bidirectional_rnn/bw/bw/strided_slice:0"Ń%
	variablesĂ%Ŕ%
X
global_step:0global_step/Assignglobal_step/read:02global_step/Initializer/zeros:0
q
conv1d_0/kernel:0conv1d_0/kernel/Assignconv1d_0/kernel/read:02,conv1d_0/kernel/Initializer/random_uniform:0
`
conv1d_0/bias:0conv1d_0/bias/Assignconv1d_0/bias/read:02!conv1d_0/bias/Initializer/zeros:0
q
conv1d_1/kernel:0conv1d_1/kernel/Assignconv1d_1/kernel/read:02,conv1d_1/kernel/Initializer/random_uniform:0
`
conv1d_1/bias:0conv1d_1/bias/Assignconv1d_1/bias/read:02!conv1d_1/bias/Initializer/zeros:0
q
conv1d_2/kernel:0conv1d_2/kernel/Assignconv1d_2/kernel/read:02,conv1d_2/kernel/Initializer/random_uniform:0
`
conv1d_2/bias:0conv1d_2/bias/Assignconv1d_2/bias/read:02!conv1d_2/bias/Initializer/zeros:0
É
Grnn_classification/cell_0/bidirectional_rnn/fw/basic_lstm_cell/kernel:0Lrnn_classification/cell_0/bidirectional_rnn/fw/basic_lstm_cell/kernel/AssignLrnn_classification/cell_0/bidirectional_rnn/fw/basic_lstm_cell/kernel/read:02brnn_classification/cell_0/bidirectional_rnn/fw/basic_lstm_cell/kernel/Initializer/random_uniform:0
¸
Ernn_classification/cell_0/bidirectional_rnn/fw/basic_lstm_cell/bias:0Jrnn_classification/cell_0/bidirectional_rnn/fw/basic_lstm_cell/bias/AssignJrnn_classification/cell_0/bidirectional_rnn/fw/basic_lstm_cell/bias/read:02Wrnn_classification/cell_0/bidirectional_rnn/fw/basic_lstm_cell/bias/Initializer/zeros:0
É
Grnn_classification/cell_0/bidirectional_rnn/bw/basic_lstm_cell/kernel:0Lrnn_classification/cell_0/bidirectional_rnn/bw/basic_lstm_cell/kernel/AssignLrnn_classification/cell_0/bidirectional_rnn/bw/basic_lstm_cell/kernel/read:02brnn_classification/cell_0/bidirectional_rnn/bw/basic_lstm_cell/kernel/Initializer/random_uniform:0
¸
Ernn_classification/cell_0/bidirectional_rnn/bw/basic_lstm_cell/bias:0Jrnn_classification/cell_0/bidirectional_rnn/bw/basic_lstm_cell/bias/AssignJrnn_classification/cell_0/bidirectional_rnn/bw/basic_lstm_cell/bias/read:02Wrnn_classification/cell_0/bidirectional_rnn/bw/basic_lstm_cell/bias/Initializer/zeros:0
É
Grnn_classification/cell_1/bidirectional_rnn/fw/basic_lstm_cell/kernel:0Lrnn_classification/cell_1/bidirectional_rnn/fw/basic_lstm_cell/kernel/AssignLrnn_classification/cell_1/bidirectional_rnn/fw/basic_lstm_cell/kernel/read:02brnn_classification/cell_1/bidirectional_rnn/fw/basic_lstm_cell/kernel/Initializer/random_uniform:0
¸
Ernn_classification/cell_1/bidirectional_rnn/fw/basic_lstm_cell/bias:0Jrnn_classification/cell_1/bidirectional_rnn/fw/basic_lstm_cell/bias/AssignJrnn_classification/cell_1/bidirectional_rnn/fw/basic_lstm_cell/bias/read:02Wrnn_classification/cell_1/bidirectional_rnn/fw/basic_lstm_cell/bias/Initializer/zeros:0
É
Grnn_classification/cell_1/bidirectional_rnn/bw/basic_lstm_cell/kernel:0Lrnn_classification/cell_1/bidirectional_rnn/bw/basic_lstm_cell/kernel/AssignLrnn_classification/cell_1/bidirectional_rnn/bw/basic_lstm_cell/kernel/read:02brnn_classification/cell_1/bidirectional_rnn/bw/basic_lstm_cell/kernel/Initializer/random_uniform:0
¸
Ernn_classification/cell_1/bidirectional_rnn/bw/basic_lstm_cell/bias:0Jrnn_classification/cell_1/bidirectional_rnn/bw/basic_lstm_cell/bias/AssignJrnn_classification/cell_1/bidirectional_rnn/bw/basic_lstm_cell/bias/read:02Wrnn_classification/cell_1/bidirectional_rnn/bw/basic_lstm_cell/bias/Initializer/zeros:0
É
Grnn_classification/cell_2/bidirectional_rnn/fw/basic_lstm_cell/kernel:0Lrnn_classification/cell_2/bidirectional_rnn/fw/basic_lstm_cell/kernel/AssignLrnn_classification/cell_2/bidirectional_rnn/fw/basic_lstm_cell/kernel/read:02brnn_classification/cell_2/bidirectional_rnn/fw/basic_lstm_cell/kernel/Initializer/random_uniform:0
¸
Ernn_classification/cell_2/bidirectional_rnn/fw/basic_lstm_cell/bias:0Jrnn_classification/cell_2/bidirectional_rnn/fw/basic_lstm_cell/bias/AssignJrnn_classification/cell_2/bidirectional_rnn/fw/basic_lstm_cell/bias/read:02Wrnn_classification/cell_2/bidirectional_rnn/fw/basic_lstm_cell/bias/Initializer/zeros:0
É
Grnn_classification/cell_2/bidirectional_rnn/bw/basic_lstm_cell/kernel:0Lrnn_classification/cell_2/bidirectional_rnn/bw/basic_lstm_cell/kernel/AssignLrnn_classification/cell_2/bidirectional_rnn/bw/basic_lstm_cell/kernel/read:02brnn_classification/cell_2/bidirectional_rnn/bw/basic_lstm_cell/kernel/Initializer/random_uniform:0
¸
Ernn_classification/cell_2/bidirectional_rnn/bw/basic_lstm_cell/bias:0Jrnn_classification/cell_2/bidirectional_rnn/bw/basic_lstm_cell/bias/AssignJrnn_classification/cell_2/bidirectional_rnn/bw/basic_lstm_cell/bias/read:02Wrnn_classification/cell_2/bidirectional_rnn/bw/basic_lstm_cell/bias/Initializer/zeros:0
e
dense/kernel:0dense/kernel/Assigndense/kernel/read:02)dense/kernel/Initializer/random_uniform:0
T
dense/bias:0dense/bias/Assigndense/bias/read:02dense/bias/Initializer/zeros:0*Ľ
outputs
3
inputs)
input_example_tensor:0˙˙˙˙˙˙˙˙˙(
scores
dense/BiasAdd:0	Ü
classes
Cast:0tensorflow/serving/classify*­
serving_default
3
inputs)
input_example_tensor:0˙˙˙˙˙˙˙˙˙
classes
Cast:0(
scores
dense/BiasAdd:0	Ütensorflow/serving/classify