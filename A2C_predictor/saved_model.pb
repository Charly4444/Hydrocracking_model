еО
шИ
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 

BiasAdd

value"T	
bias"T
output"T""
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
$
DisableCopyOnRead
resource
.
Identity

input"T
output"T"	
Ttype
u
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
2	

MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( 

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
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
H
ShardedFilename
basename	
shard

num_shards
filename
С
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring Ј
@
StaticRegexFullMatch	
input

output
"
patternstring
L

StringJoin
inputs*N

output"

Nint("
	separatorstring 
-
Tanh
x"T
y"T"
Ttype:

2
А
VarHandleOp
resource"
	containerstring "
shared_namestring "

debug_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.15.02v2.15.0-rc1-8-g6887368d6d48ур
Ш
actor_critic_model/dense_2/biasVarHandleOp*
_output_shapes
: *0

debug_name" actor_critic_model/dense_2/bias/*
dtype0*
shape:*0
shared_name!actor_critic_model/dense_2/bias

3actor_critic_model/dense_2/bias/Read/ReadVariableOpReadVariableOpactor_critic_model/dense_2/bias*
_output_shapes
:*
dtype0
в
!actor_critic_model/dense_2/kernelVarHandleOp*
_output_shapes
: *2

debug_name$"actor_critic_model/dense_2/kernel/*
dtype0*
shape
:@*2
shared_name#!actor_critic_model/dense_2/kernel

5actor_critic_model/dense_2/kernel/Read/ReadVariableOpReadVariableOp!actor_critic_model/dense_2/kernel*
_output_shapes

:@*
dtype0
Ш
actor_critic_model/dense_1/biasVarHandleOp*
_output_shapes
: *0

debug_name" actor_critic_model/dense_1/bias/*
dtype0*
shape:*0
shared_name!actor_critic_model/dense_1/bias

3actor_critic_model/dense_1/bias/Read/ReadVariableOpReadVariableOpactor_critic_model/dense_1/bias*
_output_shapes
:*
dtype0
в
!actor_critic_model/dense_1/kernelVarHandleOp*
_output_shapes
: *2

debug_name$"actor_critic_model/dense_1/kernel/*
dtype0*
shape
:@*2
shared_name#!actor_critic_model/dense_1/kernel

5actor_critic_model/dense_1/kernel/Read/ReadVariableOpReadVariableOp!actor_critic_model/dense_1/kernel*
_output_shapes

:@*
dtype0
Т
actor_critic_model/dense/biasVarHandleOp*
_output_shapes
: *.

debug_name actor_critic_model/dense/bias/*
dtype0*
shape:@*.
shared_nameactor_critic_model/dense/bias

1actor_critic_model/dense/bias/Read/ReadVariableOpReadVariableOpactor_critic_model/dense/bias*
_output_shapes
:@*
dtype0
Ь
actor_critic_model/dense/kernelVarHandleOp*
_output_shapes
: *0

debug_name" actor_critic_model/dense/kernel/*
dtype0*
shape
:@*0
shared_name!actor_critic_model/dense/kernel

3actor_critic_model/dense/kernel/Read/ReadVariableOpReadVariableOpactor_critic_model/dense/kernel*
_output_shapes

:@*
dtype0
z
serving_default_input_1Placeholder*'
_output_shapes
:џџџџџџџџџ*
dtype0*
shape:џџџџџџџџџ
Ћ
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1actor_critic_model/dense/kernelactor_critic_model/dense/bias!actor_critic_model/dense_1/kernelactor_critic_model/dense_1/bias!actor_critic_model/dense_2/kernelactor_critic_model/dense_2/bias*
Tin
	2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:џџџџџџџџџ:џџџџџџџџџ*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *>
f9R7
5__inference_signature_wrapper_serving_function_306460

NoOpNoOp

ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*е
valueЫBШ BС
ю
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

dense1
	actor_output

critic_output

signatures*
.
0
1
2
3
4
5*
.
0
1
2
3
4
5*
* 
А
non_trainable_variables

layers
metrics
layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

trace_0* 

trace_0* 
* 
І
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias*
І
	variables
 trainable_variables
!regularization_losses
"	keras_api
#__call__
*$&call_and_return_all_conditional_losses

kernel
bias*
І
%	variables
&trainable_variables
'regularization_losses
(	keras_api
)__call__
**&call_and_return_all_conditional_losses

kernel
bias*

+serving_default* 
_Y
VARIABLE_VALUEactor_critic_model/dense/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEactor_critic_model/dense/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUE!actor_critic_model/dense_1/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEactor_critic_model/dense_1/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUE!actor_critic_model/dense_2/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEactor_critic_model/dense_2/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
* 

0
	1

2*
* 
* 
* 
* 
* 

0
1*

0
1*
* 

,non_trainable_variables

-layers
.metrics
/layer_regularization_losses
0layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

1trace_0* 

2trace_0* 

0
1*

0
1*
* 

3non_trainable_variables

4layers
5metrics
6layer_regularization_losses
7layer_metrics
	variables
 trainable_variables
!regularization_losses
#__call__
*$&call_and_return_all_conditional_losses
&$"call_and_return_conditional_losses*

8trace_0* 

9trace_0* 

0
1*

0
1*
* 

:non_trainable_variables

;layers
<metrics
=layer_regularization_losses
>layer_metrics
%	variables
&trainable_variables
'regularization_losses
)__call__
**&call_and_return_all_conditional_losses
&*"call_and_return_conditional_losses*

?trace_0* 

@trace_0* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
щ
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameactor_critic_model/dense/kernelactor_critic_model/dense/bias!actor_critic_model/dense_1/kernelactor_critic_model/dense_1/bias!actor_critic_model/dense_2/kernelactor_critic_model/dense_2/biasConst*
Tin

2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *(
f#R!
__inference__traced_save_306704
ф
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameactor_critic_model/dense/kernelactor_critic_model/dense/bias!actor_critic_model/dense_1/kernelactor_critic_model/dense_1/bias!actor_critic_model/dense_2/kernelactor_critic_model/dense_2/bias*
Tin
	2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *+
f&R$
"__inference__traced_restore_306731Њ
і	
є
C__inference_dense_2_layer_call_and_return_conditional_losses_306531

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџS
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
ш

&__inference_dense_layer_call_fn_306594

inputs
unknown:@
	unknown_0:@
identityЂStatefulPartitionedCallж
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_306500o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ@<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ: : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name306590:&"
 
_user_specified_name306588:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ё
ђ
A__inference_dense_layer_call_and_return_conditional_losses_306606

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpU
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:џџџџџџџџџt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0k
MatMulMatMulCast:y:0MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ@S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ь

(__inference_dense_1_layer_call_fn_306615

inputs
unknown:@
	unknown_0:
identityЂStatefulPartitionedCallи
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_306516o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ@: : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name306611:&"
 
_user_specified_name306609:O K
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
Ю

N__inference_actor_critic_model_layer_call_and_return_conditional_losses_306539
input_1
dense_306501:@
dense_306503:@ 
dense_1_306517:@
dense_1_306519: 
dense_2_306532:@
dense_2_306534:
identity

identity_1Ђdense/StatefulPartitionedCallЂdense_1/StatefulPartitionedCallЂdense_2/StatefulPartitionedCallх
dense/StatefulPartitionedCallStatefulPartitionedCallinput_1dense_306501dense_306503*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_306500
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_306517dense_1_306519*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_306516
dense_2/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_2_306532dense_2_306534*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_306531w
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџy

Identity_1Identity(dense_2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall*
_output_shapes
 "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ: : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall:&"
 
_user_specified_name306534:&"
 
_user_specified_name306532:&"
 
_user_specified_name306519:&"
 
_user_specified_name306517:&"
 
_user_specified_name306503:&"
 
_user_specified_name306501:P L
'
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	input_1
Р

є
C__inference_dense_1_layer_call_and_return_conditional_losses_306516

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџP
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџW
IdentityIdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџS
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
і	
є
C__inference_dense_2_layer_call_and_return_conditional_losses_306645

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџS
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
ь

(__inference_dense_2_layer_call_fn_306635

inputs
unknown:@
	unknown_0:
identityЂStatefulPartitionedCallи
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_306531o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ@: : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name306631:&"
 
_user_specified_name306629:O K
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
Р

є
C__inference_dense_1_layer_call_and_return_conditional_losses_306626

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџP
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџW
IdentityIdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџS
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
Є>
з
__inference__traced_save_306704
file_prefixH
6read_disablecopyonread_actor_critic_model_dense_kernel:@D
6read_1_disablecopyonread_actor_critic_model_dense_bias:@L
:read_2_disablecopyonread_actor_critic_model_dense_1_kernel:@F
8read_3_disablecopyonread_actor_critic_model_dense_1_bias:L
:read_4_disablecopyonread_actor_critic_model_dense_2_kernel:@F
8read_5_disablecopyonread_actor_critic_model_dense_2_bias:
savev2_const
identity_13ЂMergeV2CheckpointsЂRead/DisableCopyOnReadЂRead/ReadVariableOpЂRead_1/DisableCopyOnReadЂRead_1/ReadVariableOpЂRead_2/DisableCopyOnReadЂRead_2/ReadVariableOpЂRead_3/DisableCopyOnReadЂRead_3/ReadVariableOpЂRead_4/DisableCopyOnReadЂRead_4/ReadVariableOpЂRead_5/DisableCopyOnReadЂRead_5/ReadVariableOpw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 
Read/DisableCopyOnReadDisableCopyOnRead6read_disablecopyonread_actor_critic_model_dense_kernel"/device:CPU:0*
_output_shapes
 В
Read/ReadVariableOpReadVariableOp6read_disablecopyonread_actor_critic_model_dense_kernel^Read/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@*
dtype0i
IdentityIdentityRead/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@a

Identity_1IdentityIdentity:output:0"/device:CPU:0*
T0*
_output_shapes

:@
Read_1/DisableCopyOnReadDisableCopyOnRead6read_1_disablecopyonread_actor_critic_model_dense_bias"/device:CPU:0*
_output_shapes
 В
Read_1/ReadVariableOpReadVariableOp6read_1_disablecopyonread_actor_critic_model_dense_bias^Read_1/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0i

Identity_2IdentityRead_1/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@_

Identity_3IdentityIdentity_2:output:0"/device:CPU:0*
T0*
_output_shapes
:@
Read_2/DisableCopyOnReadDisableCopyOnRead:read_2_disablecopyonread_actor_critic_model_dense_1_kernel"/device:CPU:0*
_output_shapes
 К
Read_2/ReadVariableOpReadVariableOp:read_2_disablecopyonread_actor_critic_model_dense_1_kernel^Read_2/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@*
dtype0m

Identity_4IdentityRead_2/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@c

Identity_5IdentityIdentity_4:output:0"/device:CPU:0*
T0*
_output_shapes

:@
Read_3/DisableCopyOnReadDisableCopyOnRead8read_3_disablecopyonread_actor_critic_model_dense_1_bias"/device:CPU:0*
_output_shapes
 Д
Read_3/ReadVariableOpReadVariableOp8read_3_disablecopyonread_actor_critic_model_dense_1_bias^Read_3/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0i

Identity_6IdentityRead_3/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:_

Identity_7IdentityIdentity_6:output:0"/device:CPU:0*
T0*
_output_shapes
:
Read_4/DisableCopyOnReadDisableCopyOnRead:read_4_disablecopyonread_actor_critic_model_dense_2_kernel"/device:CPU:0*
_output_shapes
 К
Read_4/ReadVariableOpReadVariableOp:read_4_disablecopyonread_actor_critic_model_dense_2_kernel^Read_4/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@*
dtype0m

Identity_8IdentityRead_4/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@c

Identity_9IdentityIdentity_8:output:0"/device:CPU:0*
T0*
_output_shapes

:@
Read_5/DisableCopyOnReadDisableCopyOnRead8read_5_disablecopyonread_actor_critic_model_dense_2_bias"/device:CPU:0*
_output_shapes
 Д
Read_5/ReadVariableOpReadVariableOp8read_5_disablecopyonread_actor_critic_model_dense_2_bias^Read_5/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0j
Identity_10IdentityRead_5/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_11IdentityIdentity_10:output:0"/device:CPU:0*
T0*
_output_shapes
:њ
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Ѓ
valueBB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH{
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*!
valueBB B B B B B B н
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Identity_1:output:0Identity_3:output:0Identity_5:output:0Identity_7:output:0Identity_9:output:0Identity_11:output:0savev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtypes
	2
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:Г
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 i
Identity_12Identityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: U
Identity_13IdentityIdentity_12:output:0^NoOp*
T0*
_output_shapes
: х
NoOpNoOp^MergeV2Checkpoints^Read/DisableCopyOnRead^Read/ReadVariableOp^Read_1/DisableCopyOnRead^Read_1/ReadVariableOp^Read_2/DisableCopyOnRead^Read_2/ReadVariableOp^Read_3/DisableCopyOnRead^Read_3/ReadVariableOp^Read_4/DisableCopyOnRead^Read_4/ReadVariableOp^Read_5/DisableCopyOnRead^Read_5/ReadVariableOp*
_output_shapes
 "#
identity_13Identity_13:output:0*(
_construction_contextkEagerRuntime*#
_input_shapes
: : : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints20
Read/DisableCopyOnReadRead/DisableCopyOnRead2*
Read/ReadVariableOpRead/ReadVariableOp24
Read_1/DisableCopyOnReadRead_1/DisableCopyOnRead2.
Read_1/ReadVariableOpRead_1/ReadVariableOp24
Read_2/DisableCopyOnReadRead_2/DisableCopyOnRead2.
Read_2/ReadVariableOpRead_2/ReadVariableOp24
Read_3/DisableCopyOnReadRead_3/DisableCopyOnRead2.
Read_3/ReadVariableOpRead_3/ReadVariableOp24
Read_4/DisableCopyOnReadRead_4/DisableCopyOnRead2.
Read_4/ReadVariableOpRead_4/ReadVariableOp24
Read_5/DisableCopyOnReadRead_5/DisableCopyOnRead2.
Read_5/ReadVariableOpRead_5/ReadVariableOp:=9

_output_shapes
: 

_user_specified_nameConst:?;
9
_user_specified_name!actor_critic_model/dense_2/bias:A=
;
_user_specified_name#!actor_critic_model/dense_2/kernel:?;
9
_user_specified_name!actor_critic_model/dense_1/bias:A=
;
_user_specified_name#!actor_critic_model/dense_1/kernel:=9
7
_user_specified_nameactor_critic_model/dense/bias:?;
9
_user_specified_name!actor_critic_model/dense/kernel:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix


3__inference_actor_critic_model_layer_call_fn_306558
input_1
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:
	unknown_3:@
	unknown_4:
identity

identity_1ЂStatefulPartitionedCallЌ
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:џџџџџџџџџ:џџџџџџџџџ*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *W
fRRP
N__inference_actor_critic_model_layer_call_and_return_conditional_losses_306539o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџq

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name306552:&"
 
_user_specified_name306550:&"
 
_user_specified_name306548:&"
 
_user_specified_name306546:&"
 
_user_specified_name306544:&"
 
_user_specified_name306542:P L
'
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	input_1
ќ'
У
!__inference__wrapped_model_306486
input_1I
7actor_critic_model_dense_matmul_readvariableop_resource:@F
8actor_critic_model_dense_biasadd_readvariableop_resource:@K
9actor_critic_model_dense_1_matmul_readvariableop_resource:@H
:actor_critic_model_dense_1_biasadd_readvariableop_resource:K
9actor_critic_model_dense_2_matmul_readvariableop_resource:@H
:actor_critic_model_dense_2_biasadd_readvariableop_resource:
identity

identity_1Ђ/actor_critic_model/dense/BiasAdd/ReadVariableOpЂ.actor_critic_model/dense/MatMul/ReadVariableOpЂ1actor_critic_model/dense_1/BiasAdd/ReadVariableOpЂ0actor_critic_model/dense_1/MatMul/ReadVariableOpЂ1actor_critic_model/dense_2/BiasAdd/ReadVariableOpЂ0actor_critic_model/dense_2/MatMul/ReadVariableOpo
actor_critic_model/dense/CastCastinput_1*

DstT0*

SrcT0*'
_output_shapes
:џџџџџџџџџІ
.actor_critic_model/dense/MatMul/ReadVariableOpReadVariableOp7actor_critic_model_dense_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0Ж
actor_critic_model/dense/MatMulMatMul!actor_critic_model/dense/Cast:y:06actor_critic_model/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@Є
/actor_critic_model/dense/BiasAdd/ReadVariableOpReadVariableOp8actor_critic_model_dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0С
 actor_critic_model/dense/BiasAddBiasAdd)actor_critic_model/dense/MatMul:product:07actor_critic_model/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@
actor_critic_model/dense/ReluRelu)actor_critic_model/dense/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@Њ
0actor_critic_model/dense_1/MatMul/ReadVariableOpReadVariableOp9actor_critic_model_dense_1_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0Ф
!actor_critic_model/dense_1/MatMulMatMul+actor_critic_model/dense/Relu:activations:08actor_critic_model/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџЈ
1actor_critic_model/dense_1/BiasAdd/ReadVariableOpReadVariableOp:actor_critic_model_dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ч
"actor_critic_model/dense_1/BiasAddBiasAdd+actor_critic_model/dense_1/MatMul:product:09actor_critic_model/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
actor_critic_model/dense_1/TanhTanh+actor_critic_model/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџЊ
0actor_critic_model/dense_2/MatMul/ReadVariableOpReadVariableOp9actor_critic_model_dense_2_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0Ф
!actor_critic_model/dense_2/MatMulMatMul+actor_critic_model/dense/Relu:activations:08actor_critic_model/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџЈ
1actor_critic_model/dense_2/BiasAdd/ReadVariableOpReadVariableOp:actor_critic_model_dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ч
"actor_critic_model/dense_2/BiasAddBiasAdd+actor_critic_model/dense_2/MatMul:product:09actor_critic_model/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџr
IdentityIdentity#actor_critic_model/dense_1/Tanh:y:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ|

Identity_1Identity+actor_critic_model/dense_2/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџг
NoOpNoOp0^actor_critic_model/dense/BiasAdd/ReadVariableOp/^actor_critic_model/dense/MatMul/ReadVariableOp2^actor_critic_model/dense_1/BiasAdd/ReadVariableOp1^actor_critic_model/dense_1/MatMul/ReadVariableOp2^actor_critic_model/dense_2/BiasAdd/ReadVariableOp1^actor_critic_model/dense_2/MatMul/ReadVariableOp*
_output_shapes
 "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ: : : : : : 2b
/actor_critic_model/dense/BiasAdd/ReadVariableOp/actor_critic_model/dense/BiasAdd/ReadVariableOp2`
.actor_critic_model/dense/MatMul/ReadVariableOp.actor_critic_model/dense/MatMul/ReadVariableOp2f
1actor_critic_model/dense_1/BiasAdd/ReadVariableOp1actor_critic_model/dense_1/BiasAdd/ReadVariableOp2d
0actor_critic_model/dense_1/MatMul/ReadVariableOp0actor_critic_model/dense_1/MatMul/ReadVariableOp2f
1actor_critic_model/dense_2/BiasAdd/ReadVariableOp1actor_critic_model/dense_2/BiasAdd/ReadVariableOp2d
0actor_critic_model/dense_2/MatMul/ReadVariableOp0actor_critic_model/dense_2/MatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:P L
'
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	input_1
#
м
"__inference__traced_restore_306731
file_prefixB
0assignvariableop_actor_critic_model_dense_kernel:@>
0assignvariableop_1_actor_critic_model_dense_bias:@F
4assignvariableop_2_actor_critic_model_dense_1_kernel:@@
2assignvariableop_3_actor_critic_model_dense_1_bias:F
4assignvariableop_4_actor_critic_model_dense_2_kernel:@@
2assignvariableop_5_actor_critic_model_dense_2_bias:

identity_7ЂAssignVariableOpЂAssignVariableOp_1ЂAssignVariableOp_2ЂAssignVariableOp_3ЂAssignVariableOp_4ЂAssignVariableOp_5§
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Ѓ
valueBB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH~
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*!
valueBB B B B B B B С
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*0
_output_shapes
:::::::*
dtypes
	2[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:У
AssignVariableOpAssignVariableOp0assignvariableop_actor_critic_model_dense_kernelIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:Ч
AssignVariableOp_1AssignVariableOp0assignvariableop_1_actor_critic_model_dense_biasIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:Ы
AssignVariableOp_2AssignVariableOp4assignvariableop_2_actor_critic_model_dense_1_kernelIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:Щ
AssignVariableOp_3AssignVariableOp2assignvariableop_3_actor_critic_model_dense_1_biasIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:Ы
AssignVariableOp_4AssignVariableOp4assignvariableop_4_actor_critic_model_dense_2_kernelIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:Щ
AssignVariableOp_5AssignVariableOp2assignvariableop_5_actor_critic_model_dense_2_biasIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 ж

Identity_6Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^NoOp"/device:CPU:0*
T0*
_output_shapes
: U

Identity_7IdentityIdentity_6:output:0^NoOp_1*
T0*
_output_shapes
:  
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5*
_output_shapes
 "!

identity_7Identity_7:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
: : : : : : : 2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52$
AssignVariableOpAssignVariableOp:?;
9
_user_specified_name!actor_critic_model/dense_2/bias:A=
;
_user_specified_name#!actor_critic_model/dense_2/kernel:?;
9
_user_specified_name!actor_critic_model/dense_1/bias:A=
;
_user_specified_name#!actor_critic_model/dense_1/kernel:=9
7
_user_specified_nameactor_critic_model/dense/bias:?;
9
_user_specified_name!actor_critic_model/dense/kernel:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
Ё
ђ
A__inference_dense_layer_call_and_return_conditional_losses_306500

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpU
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:џџџџџџџџџt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0k
MatMulMatMulCast:y:0MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ@S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
о

5__inference_signature_wrapper_serving_function_306460
input_1
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:
	unknown_3:@
	unknown_4:
identity

identity_1ЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:џџџџџџџџџ:џџџџџџџџџ*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *,
f'R%
#__inference_serving_function_306440o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџq

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name306454:&"
 
_user_specified_name306452:&"
 
_user_specified_name306450:&"
 
_user_specified_name306448:&"
 
_user_specified_name306446:&"
 
_user_specified_name306444:P L
'
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	input_1

с
#__inference_serving_function_306440
input_16
$dense_matmul_readvariableop_resource:@3
%dense_biasadd_readvariableop_resource:@8
&dense_1_matmul_readvariableop_resource:@5
'dense_1_biasadd_readvariableop_resource:8
&dense_2_matmul_readvariableop_resource:@5
'dense_2_biasadd_readvariableop_resource:
identity

identity_1Ђdense/BiasAdd/ReadVariableOpЂdense/MatMul/ReadVariableOpЂdense_1/BiasAdd/ReadVariableOpЂdense_1/MatMul/ReadVariableOpЂdense_2/BiasAdd/ReadVariableOpЂdense_2/MatMul/ReadVariableOp
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0v
dense/MatMulMatMulinput_1#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@\

dense/ReluReludense/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0
dense_1/MatMulMatMuldense/Relu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ`
dense_1/TanhTanhdense_1/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0
dense_2/MatMulMatMuldense/Relu:activations:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ_
IdentityIdentitydense_1/Tanh:y:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџi

Identity_1Identitydense_2/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџс
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp*
_output_shapes
 "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ: : : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:P L
'
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	input_1"эL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*щ
serving_defaultе
;
input_10
serving_default_input_1:0џџџџџџџџџ<
output_00
StatefulPartitionedCall:0џџџџџџџџџ<
output_10
StatefulPartitionedCall:1џџџџџџџџџtensorflow/serving/predict:хK

	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

dense1
	actor_output

critic_output

signatures"
_tf_keras_model
J
0
1
2
3
4
5"
trackable_list_wrapper
J
0
1
2
3
4
5"
trackable_list_wrapper
 "
trackable_list_wrapper
Ъ
non_trainable_variables

layers
metrics
layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
э
trace_02а
3__inference_actor_critic_model_layer_call_fn_306558
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0

trace_02ы
N__inference_actor_critic_model_layer_call_and_return_conditional_losses_306539
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0
ЬBЩ
!__inference__wrapped_model_306486input_1"
В
FullArgSpec
args

jargs_0
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Л
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
Л
	variables
 trainable_variables
!regularization_losses
"	keras_api
#__call__
*$&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
Л
%	variables
&trainable_variables
'regularization_losses
(	keras_api
)__call__
**&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
,
+serving_default"
signature_map
1:/@2actor_critic_model/dense/kernel
+:)@2actor_critic_model/dense/bias
3:1@2!actor_critic_model/dense_1/kernel
-:+2actor_critic_model/dense_1/bias
3:1@2!actor_critic_model/dense_2/kernel
-:+2actor_critic_model/dense_2/bias
 "
trackable_list_wrapper
5
0
	1

2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
оBл
3__inference_actor_critic_model_layer_call_fn_306558input_1"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
љBі
N__inference_actor_critic_model_layer_call_and_return_conditional_losses_306539input_1"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
,non_trainable_variables

-layers
.metrics
/layer_regularization_losses
0layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
р
1trace_02У
&__inference_dense_layer_call_fn_306594
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 z1trace_0
ћ
2trace_02о
A__inference_dense_layer_call_and_return_conditional_losses_306606
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 z2trace_0
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
3non_trainable_variables

4layers
5metrics
6layer_regularization_losses
7layer_metrics
	variables
 trainable_variables
!regularization_losses
#__call__
*$&call_and_return_all_conditional_losses
&$"call_and_return_conditional_losses"
_generic_user_object
т
8trace_02Х
(__inference_dense_1_layer_call_fn_306615
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 z8trace_0
§
9trace_02р
C__inference_dense_1_layer_call_and_return_conditional_losses_306626
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 z9trace_0
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
:non_trainable_variables

;layers
<metrics
=layer_regularization_losses
>layer_metrics
%	variables
&trainable_variables
'regularization_losses
)__call__
**&call_and_return_all_conditional_losses
&*"call_and_return_conditional_losses"
_generic_user_object
т
?trace_02Х
(__inference_dense_2_layer_call_fn_306635
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 z?trace_0
§
@trace_02р
C__inference_dense_2_layer_call_and_return_conditional_losses_306645
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 z@trace_0
сBо
5__inference_signature_wrapper_serving_function_306460input_1"
В
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs
	jinput_1
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
аBЭ
&__inference_dense_layer_call_fn_306594inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ыBш
A__inference_dense_layer_call_and_return_conditional_losses_306606inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
вBЯ
(__inference_dense_1_layer_call_fn_306615inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
эBъ
C__inference_dense_1_layer_call_and_return_conditional_losses_306626inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
вBЯ
(__inference_dense_2_layer_call_fn_306635inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
эBъ
C__inference_dense_2_layer_call_and_return_conditional_losses_306645inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 Х
!__inference__wrapped_model_3064860Ђ-
&Ђ#
!
input_1џџџџџџџџџ
Њ "cЊ`
.
output_1"
output_1џџџџџџџџџ
.
output_2"
output_2џџџџџџџџџш
N__inference_actor_critic_model_layer_call_and_return_conditional_losses_3065390Ђ-
&Ђ#
!
input_1џџџџџџџџџ
Њ "YЂV
OЂL
$!

tensor_0_0џџџџџџџџџ
$!

tensor_0_1џџџџџџџџџ
 П
3__inference_actor_critic_model_layer_call_fn_3065580Ђ-
&Ђ#
!
input_1џџџџџџџџџ
Њ "KЂH
"
tensor_0џџџџџџџџџ
"
tensor_1џџџџџџџџџЊ
C__inference_dense_1_layer_call_and_return_conditional_losses_306626c/Ђ,
%Ђ"
 
inputsџџџџџџџџџ@
Њ ",Ђ)
"
tensor_0џџџџџџџџџ
 
(__inference_dense_1_layer_call_fn_306615X/Ђ,
%Ђ"
 
inputsџџџџџџџџџ@
Њ "!
unknownџџџџџџџџџЊ
C__inference_dense_2_layer_call_and_return_conditional_losses_306645c/Ђ,
%Ђ"
 
inputsџџџџџџџџџ@
Њ ",Ђ)
"
tensor_0џџџџџџџџџ
 
(__inference_dense_2_layer_call_fn_306635X/Ђ,
%Ђ"
 
inputsџџџџџџџџџ@
Њ "!
unknownџџџџџџџџџЈ
A__inference_dense_layer_call_and_return_conditional_losses_306606c/Ђ,
%Ђ"
 
inputsџџџџџџџџџ
Њ ",Ђ)
"
tensor_0џџџџџџџџџ@
 
&__inference_dense_layer_call_fn_306594X/Ђ,
%Ђ"
 
inputsџџџџџџџџџ
Њ "!
unknownџџџџџџџџџ@ф
5__inference_signature_wrapper_serving_function_306460Њ;Ђ8
Ђ 
1Њ.
,
input_1!
input_1џџџџџџџџџ"cЊ`
.
output_0"
output_0џџџџџџџџџ
.
output_1"
output_1џџџџџџџџџ