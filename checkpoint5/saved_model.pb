Æ¥
¿£
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
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
¾
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
executor_typestring 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.3.02unknown8íñ

le_net_fcb_n3/conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_namele_net_fcb_n3/conv2d/kernel

/le_net_fcb_n3/conv2d/kernel/Read/ReadVariableOpReadVariableOple_net_fcb_n3/conv2d/kernel*&
_output_shapes
: *
dtype0

le_net_fcb_n3/conv2d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: **
shared_namele_net_fcb_n3/conv2d/bias

-le_net_fcb_n3/conv2d/bias/Read/ReadVariableOpReadVariableOple_net_fcb_n3/conv2d/bias*
_output_shapes
: *
dtype0
¦
'le_net_fcb_n3/batch_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *8
shared_name)'le_net_fcb_n3/batch_normalization/gamma

;le_net_fcb_n3/batch_normalization/gamma/Read/ReadVariableOpReadVariableOp'le_net_fcb_n3/batch_normalization/gamma*
_output_shapes
: *
dtype0
¤
&le_net_fcb_n3/batch_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *7
shared_name(&le_net_fcb_n3/batch_normalization/beta

:le_net_fcb_n3/batch_normalization/beta/Read/ReadVariableOpReadVariableOp&le_net_fcb_n3/batch_normalization/beta*
_output_shapes
: *
dtype0
²
-le_net_fcb_n3/batch_normalization/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *>
shared_name/-le_net_fcb_n3/batch_normalization/moving_mean
«
Ale_net_fcb_n3/batch_normalization/moving_mean/Read/ReadVariableOpReadVariableOp-le_net_fcb_n3/batch_normalization/moving_mean*
_output_shapes
: *
dtype0
º
1le_net_fcb_n3/batch_normalization/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *B
shared_name31le_net_fcb_n3/batch_normalization/moving_variance
³
Ele_net_fcb_n3/batch_normalization/moving_variance/Read/ReadVariableOpReadVariableOp1le_net_fcb_n3/batch_normalization/moving_variance*
_output_shapes
: *
dtype0

le_net_fcb_n3/conv2d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*.
shared_namele_net_fcb_n3/conv2d_1/kernel

1le_net_fcb_n3/conv2d_1/kernel/Read/ReadVariableOpReadVariableOple_net_fcb_n3/conv2d_1/kernel*&
_output_shapes
: @*
dtype0

le_net_fcb_n3/conv2d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namele_net_fcb_n3/conv2d_1/bias

/le_net_fcb_n3/conv2d_1/bias/Read/ReadVariableOpReadVariableOple_net_fcb_n3/conv2d_1/bias*
_output_shapes
:@*
dtype0

le_net_fcb_n3/conv2d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*.
shared_namele_net_fcb_n3/conv2d_2/kernel

1le_net_fcb_n3/conv2d_2/kernel/Read/ReadVariableOpReadVariableOple_net_fcb_n3/conv2d_2/kernel*'
_output_shapes
:@*
dtype0

le_net_fcb_n3/conv2d_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namele_net_fcb_n3/conv2d_2/bias

/le_net_fcb_n3/conv2d_2/bias/Read/ReadVariableOpReadVariableOple_net_fcb_n3/conv2d_2/bias*
_output_shapes	
:*
dtype0

le_net_fcb_n3/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ð*+
shared_namele_net_fcb_n3/dense/kernel

.le_net_fcb_n3/dense/kernel/Read/ReadVariableOpReadVariableOple_net_fcb_n3/dense/kernel* 
_output_shapes
:
ð*
dtype0

le_net_fcb_n3/dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:ð*)
shared_namele_net_fcb_n3/dense/bias

,le_net_fcb_n3/dense/bias/Read/ReadVariableOpReadVariableOple_net_fcb_n3/dense/bias*
_output_shapes	
:ð*
dtype0
«
)le_net_fcb_n3/batch_normalization_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:ð*:
shared_name+)le_net_fcb_n3/batch_normalization_1/gamma
¤
=le_net_fcb_n3/batch_normalization_1/gamma/Read/ReadVariableOpReadVariableOp)le_net_fcb_n3/batch_normalization_1/gamma*
_output_shapes	
:ð*
dtype0
©
(le_net_fcb_n3/batch_normalization_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:ð*9
shared_name*(le_net_fcb_n3/batch_normalization_1/beta
¢
<le_net_fcb_n3/batch_normalization_1/beta/Read/ReadVariableOpReadVariableOp(le_net_fcb_n3/batch_normalization_1/beta*
_output_shapes	
:ð*
dtype0
·
/le_net_fcb_n3/batch_normalization_1/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:ð*@
shared_name1/le_net_fcb_n3/batch_normalization_1/moving_mean
°
Cle_net_fcb_n3/batch_normalization_1/moving_mean/Read/ReadVariableOpReadVariableOp/le_net_fcb_n3/batch_normalization_1/moving_mean*
_output_shapes	
:ð*
dtype0
¿
3le_net_fcb_n3/batch_normalization_1/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:ð*D
shared_name53le_net_fcb_n3/batch_normalization_1/moving_variance
¸
Gle_net_fcb_n3/batch_normalization_1/moving_variance/Read/ReadVariableOpReadVariableOp3le_net_fcb_n3/batch_normalization_1/moving_variance*
_output_shapes	
:ð*
dtype0
ª
)le_net_fcb_n3/batch_normalization_2/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:T*:
shared_name+)le_net_fcb_n3/batch_normalization_2/gamma
£
=le_net_fcb_n3/batch_normalization_2/gamma/Read/ReadVariableOpReadVariableOp)le_net_fcb_n3/batch_normalization_2/gamma*
_output_shapes
:T*
dtype0
¨
(le_net_fcb_n3/batch_normalization_2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:T*9
shared_name*(le_net_fcb_n3/batch_normalization_2/beta
¡
<le_net_fcb_n3/batch_normalization_2/beta/Read/ReadVariableOpReadVariableOp(le_net_fcb_n3/batch_normalization_2/beta*
_output_shapes
:T*
dtype0
¶
/le_net_fcb_n3/batch_normalization_2/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:T*@
shared_name1/le_net_fcb_n3/batch_normalization_2/moving_mean
¯
Cle_net_fcb_n3/batch_normalization_2/moving_mean/Read/ReadVariableOpReadVariableOp/le_net_fcb_n3/batch_normalization_2/moving_mean*
_output_shapes
:T*
dtype0
¾
3le_net_fcb_n3/batch_normalization_2/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:T*D
shared_name53le_net_fcb_n3/batch_normalization_2/moving_variance
·
Gle_net_fcb_n3/batch_normalization_2/moving_variance/Read/ReadVariableOpReadVariableOp3le_net_fcb_n3/batch_normalization_2/moving_variance*
_output_shapes
:T*
dtype0

le_net_fcb_n3/dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	ðT*-
shared_namele_net_fcb_n3/dense_1/kernel

0le_net_fcb_n3/dense_1/kernel/Read/ReadVariableOpReadVariableOple_net_fcb_n3/dense_1/kernel*
_output_shapes
:	ðT*
dtype0

le_net_fcb_n3/dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:T*+
shared_namele_net_fcb_n3/dense_1/bias

.le_net_fcb_n3/dense_1/bias/Read/ReadVariableOpReadVariableOple_net_fcb_n3/dense_1/bias*
_output_shapes
:T*
dtype0

le_net_fcb_n3/dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:T*-
shared_namele_net_fcb_n3/dense_2/kernel

0le_net_fcb_n3/dense_2/kernel/Read/ReadVariableOpReadVariableOple_net_fcb_n3/dense_2/kernel*
_output_shapes

:T*
dtype0

le_net_fcb_n3/dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namele_net_fcb_n3/dense_2/bias

.le_net_fcb_n3/dense_2/bias/Read/ReadVariableOpReadVariableOple_net_fcb_n3/dense_2/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
¨
"Adam/le_net_fcb_n3/conv2d/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"Adam/le_net_fcb_n3/conv2d/kernel/m
¡
6Adam/le_net_fcb_n3/conv2d/kernel/m/Read/ReadVariableOpReadVariableOp"Adam/le_net_fcb_n3/conv2d/kernel/m*&
_output_shapes
: *
dtype0

 Adam/le_net_fcb_n3/conv2d/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *1
shared_name" Adam/le_net_fcb_n3/conv2d/bias/m

4Adam/le_net_fcb_n3/conv2d/bias/m/Read/ReadVariableOpReadVariableOp Adam/le_net_fcb_n3/conv2d/bias/m*
_output_shapes
: *
dtype0
´
.Adam/le_net_fcb_n3/batch_normalization/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *?
shared_name0.Adam/le_net_fcb_n3/batch_normalization/gamma/m
­
BAdam/le_net_fcb_n3/batch_normalization/gamma/m/Read/ReadVariableOpReadVariableOp.Adam/le_net_fcb_n3/batch_normalization/gamma/m*
_output_shapes
: *
dtype0
²
-Adam/le_net_fcb_n3/batch_normalization/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *>
shared_name/-Adam/le_net_fcb_n3/batch_normalization/beta/m
«
AAdam/le_net_fcb_n3/batch_normalization/beta/m/Read/ReadVariableOpReadVariableOp-Adam/le_net_fcb_n3/batch_normalization/beta/m*
_output_shapes
: *
dtype0
¬
$Adam/le_net_fcb_n3/conv2d_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*5
shared_name&$Adam/le_net_fcb_n3/conv2d_1/kernel/m
¥
8Adam/le_net_fcb_n3/conv2d_1/kernel/m/Read/ReadVariableOpReadVariableOp$Adam/le_net_fcb_n3/conv2d_1/kernel/m*&
_output_shapes
: @*
dtype0

"Adam/le_net_fcb_n3/conv2d_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"Adam/le_net_fcb_n3/conv2d_1/bias/m

6Adam/le_net_fcb_n3/conv2d_1/bias/m/Read/ReadVariableOpReadVariableOp"Adam/le_net_fcb_n3/conv2d_1/bias/m*
_output_shapes
:@*
dtype0
­
$Adam/le_net_fcb_n3/conv2d_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*5
shared_name&$Adam/le_net_fcb_n3/conv2d_2/kernel/m
¦
8Adam/le_net_fcb_n3/conv2d_2/kernel/m/Read/ReadVariableOpReadVariableOp$Adam/le_net_fcb_n3/conv2d_2/kernel/m*'
_output_shapes
:@*
dtype0

"Adam/le_net_fcb_n3/conv2d_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/le_net_fcb_n3/conv2d_2/bias/m

6Adam/le_net_fcb_n3/conv2d_2/bias/m/Read/ReadVariableOpReadVariableOp"Adam/le_net_fcb_n3/conv2d_2/bias/m*
_output_shapes	
:*
dtype0
 
!Adam/le_net_fcb_n3/dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ð*2
shared_name#!Adam/le_net_fcb_n3/dense/kernel/m

5Adam/le_net_fcb_n3/dense/kernel/m/Read/ReadVariableOpReadVariableOp!Adam/le_net_fcb_n3/dense/kernel/m* 
_output_shapes
:
ð*
dtype0

Adam/le_net_fcb_n3/dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:ð*0
shared_name!Adam/le_net_fcb_n3/dense/bias/m

3Adam/le_net_fcb_n3/dense/bias/m/Read/ReadVariableOpReadVariableOpAdam/le_net_fcb_n3/dense/bias/m*
_output_shapes	
:ð*
dtype0
¹
0Adam/le_net_fcb_n3/batch_normalization_1/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:ð*A
shared_name20Adam/le_net_fcb_n3/batch_normalization_1/gamma/m
²
DAdam/le_net_fcb_n3/batch_normalization_1/gamma/m/Read/ReadVariableOpReadVariableOp0Adam/le_net_fcb_n3/batch_normalization_1/gamma/m*
_output_shapes	
:ð*
dtype0
·
/Adam/le_net_fcb_n3/batch_normalization_1/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:ð*@
shared_name1/Adam/le_net_fcb_n3/batch_normalization_1/beta/m
°
CAdam/le_net_fcb_n3/batch_normalization_1/beta/m/Read/ReadVariableOpReadVariableOp/Adam/le_net_fcb_n3/batch_normalization_1/beta/m*
_output_shapes	
:ð*
dtype0
¸
0Adam/le_net_fcb_n3/batch_normalization_2/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:T*A
shared_name20Adam/le_net_fcb_n3/batch_normalization_2/gamma/m
±
DAdam/le_net_fcb_n3/batch_normalization_2/gamma/m/Read/ReadVariableOpReadVariableOp0Adam/le_net_fcb_n3/batch_normalization_2/gamma/m*
_output_shapes
:T*
dtype0
¶
/Adam/le_net_fcb_n3/batch_normalization_2/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:T*@
shared_name1/Adam/le_net_fcb_n3/batch_normalization_2/beta/m
¯
CAdam/le_net_fcb_n3/batch_normalization_2/beta/m/Read/ReadVariableOpReadVariableOp/Adam/le_net_fcb_n3/batch_normalization_2/beta/m*
_output_shapes
:T*
dtype0
£
#Adam/le_net_fcb_n3/dense_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	ðT*4
shared_name%#Adam/le_net_fcb_n3/dense_1/kernel/m

7Adam/le_net_fcb_n3/dense_1/kernel/m/Read/ReadVariableOpReadVariableOp#Adam/le_net_fcb_n3/dense_1/kernel/m*
_output_shapes
:	ðT*
dtype0

!Adam/le_net_fcb_n3/dense_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:T*2
shared_name#!Adam/le_net_fcb_n3/dense_1/bias/m

5Adam/le_net_fcb_n3/dense_1/bias/m/Read/ReadVariableOpReadVariableOp!Adam/le_net_fcb_n3/dense_1/bias/m*
_output_shapes
:T*
dtype0
¢
#Adam/le_net_fcb_n3/dense_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:T*4
shared_name%#Adam/le_net_fcb_n3/dense_2/kernel/m

7Adam/le_net_fcb_n3/dense_2/kernel/m/Read/ReadVariableOpReadVariableOp#Adam/le_net_fcb_n3/dense_2/kernel/m*
_output_shapes

:T*
dtype0

!Adam/le_net_fcb_n3/dense_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/le_net_fcb_n3/dense_2/bias/m

5Adam/le_net_fcb_n3/dense_2/bias/m/Read/ReadVariableOpReadVariableOp!Adam/le_net_fcb_n3/dense_2/bias/m*
_output_shapes
:*
dtype0
¨
"Adam/le_net_fcb_n3/conv2d/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"Adam/le_net_fcb_n3/conv2d/kernel/v
¡
6Adam/le_net_fcb_n3/conv2d/kernel/v/Read/ReadVariableOpReadVariableOp"Adam/le_net_fcb_n3/conv2d/kernel/v*&
_output_shapes
: *
dtype0

 Adam/le_net_fcb_n3/conv2d/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *1
shared_name" Adam/le_net_fcb_n3/conv2d/bias/v

4Adam/le_net_fcb_n3/conv2d/bias/v/Read/ReadVariableOpReadVariableOp Adam/le_net_fcb_n3/conv2d/bias/v*
_output_shapes
: *
dtype0
´
.Adam/le_net_fcb_n3/batch_normalization/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *?
shared_name0.Adam/le_net_fcb_n3/batch_normalization/gamma/v
­
BAdam/le_net_fcb_n3/batch_normalization/gamma/v/Read/ReadVariableOpReadVariableOp.Adam/le_net_fcb_n3/batch_normalization/gamma/v*
_output_shapes
: *
dtype0
²
-Adam/le_net_fcb_n3/batch_normalization/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *>
shared_name/-Adam/le_net_fcb_n3/batch_normalization/beta/v
«
AAdam/le_net_fcb_n3/batch_normalization/beta/v/Read/ReadVariableOpReadVariableOp-Adam/le_net_fcb_n3/batch_normalization/beta/v*
_output_shapes
: *
dtype0
¬
$Adam/le_net_fcb_n3/conv2d_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*5
shared_name&$Adam/le_net_fcb_n3/conv2d_1/kernel/v
¥
8Adam/le_net_fcb_n3/conv2d_1/kernel/v/Read/ReadVariableOpReadVariableOp$Adam/le_net_fcb_n3/conv2d_1/kernel/v*&
_output_shapes
: @*
dtype0

"Adam/le_net_fcb_n3/conv2d_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"Adam/le_net_fcb_n3/conv2d_1/bias/v

6Adam/le_net_fcb_n3/conv2d_1/bias/v/Read/ReadVariableOpReadVariableOp"Adam/le_net_fcb_n3/conv2d_1/bias/v*
_output_shapes
:@*
dtype0
­
$Adam/le_net_fcb_n3/conv2d_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*5
shared_name&$Adam/le_net_fcb_n3/conv2d_2/kernel/v
¦
8Adam/le_net_fcb_n3/conv2d_2/kernel/v/Read/ReadVariableOpReadVariableOp$Adam/le_net_fcb_n3/conv2d_2/kernel/v*'
_output_shapes
:@*
dtype0

"Adam/le_net_fcb_n3/conv2d_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/le_net_fcb_n3/conv2d_2/bias/v

6Adam/le_net_fcb_n3/conv2d_2/bias/v/Read/ReadVariableOpReadVariableOp"Adam/le_net_fcb_n3/conv2d_2/bias/v*
_output_shapes	
:*
dtype0
 
!Adam/le_net_fcb_n3/dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ð*2
shared_name#!Adam/le_net_fcb_n3/dense/kernel/v

5Adam/le_net_fcb_n3/dense/kernel/v/Read/ReadVariableOpReadVariableOp!Adam/le_net_fcb_n3/dense/kernel/v* 
_output_shapes
:
ð*
dtype0

Adam/le_net_fcb_n3/dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:ð*0
shared_name!Adam/le_net_fcb_n3/dense/bias/v

3Adam/le_net_fcb_n3/dense/bias/v/Read/ReadVariableOpReadVariableOpAdam/le_net_fcb_n3/dense/bias/v*
_output_shapes	
:ð*
dtype0
¹
0Adam/le_net_fcb_n3/batch_normalization_1/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:ð*A
shared_name20Adam/le_net_fcb_n3/batch_normalization_1/gamma/v
²
DAdam/le_net_fcb_n3/batch_normalization_1/gamma/v/Read/ReadVariableOpReadVariableOp0Adam/le_net_fcb_n3/batch_normalization_1/gamma/v*
_output_shapes	
:ð*
dtype0
·
/Adam/le_net_fcb_n3/batch_normalization_1/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:ð*@
shared_name1/Adam/le_net_fcb_n3/batch_normalization_1/beta/v
°
CAdam/le_net_fcb_n3/batch_normalization_1/beta/v/Read/ReadVariableOpReadVariableOp/Adam/le_net_fcb_n3/batch_normalization_1/beta/v*
_output_shapes	
:ð*
dtype0
¸
0Adam/le_net_fcb_n3/batch_normalization_2/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:T*A
shared_name20Adam/le_net_fcb_n3/batch_normalization_2/gamma/v
±
DAdam/le_net_fcb_n3/batch_normalization_2/gamma/v/Read/ReadVariableOpReadVariableOp0Adam/le_net_fcb_n3/batch_normalization_2/gamma/v*
_output_shapes
:T*
dtype0
¶
/Adam/le_net_fcb_n3/batch_normalization_2/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:T*@
shared_name1/Adam/le_net_fcb_n3/batch_normalization_2/beta/v
¯
CAdam/le_net_fcb_n3/batch_normalization_2/beta/v/Read/ReadVariableOpReadVariableOp/Adam/le_net_fcb_n3/batch_normalization_2/beta/v*
_output_shapes
:T*
dtype0
£
#Adam/le_net_fcb_n3/dense_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	ðT*4
shared_name%#Adam/le_net_fcb_n3/dense_1/kernel/v

7Adam/le_net_fcb_n3/dense_1/kernel/v/Read/ReadVariableOpReadVariableOp#Adam/le_net_fcb_n3/dense_1/kernel/v*
_output_shapes
:	ðT*
dtype0

!Adam/le_net_fcb_n3/dense_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:T*2
shared_name#!Adam/le_net_fcb_n3/dense_1/bias/v

5Adam/le_net_fcb_n3/dense_1/bias/v/Read/ReadVariableOpReadVariableOp!Adam/le_net_fcb_n3/dense_1/bias/v*
_output_shapes
:T*
dtype0
¢
#Adam/le_net_fcb_n3/dense_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:T*4
shared_name%#Adam/le_net_fcb_n3/dense_2/kernel/v

7Adam/le_net_fcb_n3/dense_2/kernel/v/Read/ReadVariableOpReadVariableOp#Adam/le_net_fcb_n3/dense_2/kernel/v*
_output_shapes

:T*
dtype0

!Adam/le_net_fcb_n3/dense_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/le_net_fcb_n3/dense_2/bias/v

5Adam/le_net_fcb_n3/dense_2/bias/v/Read/ReadVariableOpReadVariableOp!Adam/le_net_fcb_n3/dense_2/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
¹r
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*ôq
valueêqBçq Bàq
§
c1
affine1
s2
c3
s4
affine2
	c_add

affine_add
		s_add

flatten
c5
c5_drop
affine3
affine4
f6
output_layer
	optimizer
trainable_variables
	variables
regularization_losses
	keras_api

signatures
h

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api

axis
	gamma
beta
 moving_mean
!moving_variance
"trainable_variables
#	variables
$regularization_losses
%	keras_api
R
&trainable_variables
'	variables
(regularization_losses
)	keras_api
h

*kernel
+bias
,trainable_variables
-	variables
.regularization_losses
/	keras_api
R
0trainable_variables
1	variables
2regularization_losses
3	keras_api
R
4trainable_variables
5	variables
6regularization_losses
7	keras_api
h

8kernel
9bias
:trainable_variables
;	variables
<regularization_losses
=	keras_api
R
>trainable_variables
?	variables
@regularization_losses
A	keras_api
R
Btrainable_variables
C	variables
Dregularization_losses
E	keras_api
R
Ftrainable_variables
G	variables
Hregularization_losses
I	keras_api
h

Jkernel
Kbias
Ltrainable_variables
M	variables
Nregularization_losses
O	keras_api
R
Ptrainable_variables
Q	variables
Rregularization_losses
S	keras_api

Taxis
	Ugamma
Vbeta
Wmoving_mean
Xmoving_variance
Ytrainable_variables
Z	variables
[regularization_losses
\	keras_api

]axis
	^gamma
_beta
`moving_mean
amoving_variance
btrainable_variables
c	variables
dregularization_losses
e	keras_api
h

fkernel
gbias
htrainable_variables
i	variables
jregularization_losses
k	keras_api
h

lkernel
mbias
ntrainable_variables
o	variables
pregularization_losses
q	keras_api
¨
riter

sbeta_1

tbeta_2
	udecay
vlearning_ratem×mØmÙmÚ*mÛ+mÜ8mÝ9mÞJmßKmàUmáVmâ^mã_mäfmågmælmçmmèvévêvëvì*ví+vî8vï9vðJvñKvòUvóVvô^võ_vöfv÷gvølvùmvú

0
1
2
3
*4
+5
86
97
J8
K9
U10
V11
^12
_13
f14
g15
l16
m17
¶
0
1
2
3
 4
!5
*6
+7
88
99
J10
K11
U12
V13
W14
X15
^16
_17
`18
a19
f20
g21
l22
m23
 
­
wlayer_regularization_losses
xlayer_metrics
trainable_variables
	variables
ynon_trainable_variables
zmetrics
regularization_losses

{layers
 
US
VARIABLE_VALUEle_net_fcb_n3/conv2d/kernel$c1/kernel/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEle_net_fcb_n3/conv2d/bias"c1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
®
|layer_regularization_losses
}layer_metrics
trainable_variables
	variables
~non_trainable_variables
metrics
regularization_losses
layers
 
ec
VARIABLE_VALUE'le_net_fcb_n3/batch_normalization/gamma(affine1/gamma/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUE&le_net_fcb_n3/batch_normalization/beta'affine1/beta/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUE-le_net_fcb_n3/batch_normalization/moving_mean.affine1/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUE1le_net_fcb_n3/batch_normalization/moving_variance2affine1/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 2
!3
 
²
 layer_regularization_losses
layer_metrics
"trainable_variables
#	variables
non_trainable_variables
metrics
$regularization_losses
layers
 
 
 
²
 layer_regularization_losses
layer_metrics
&trainable_variables
'	variables
non_trainable_variables
metrics
(regularization_losses
layers
WU
VARIABLE_VALUEle_net_fcb_n3/conv2d_1/kernel$c3/kernel/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEle_net_fcb_n3/conv2d_1/bias"c3/bias/.ATTRIBUTES/VARIABLE_VALUE

*0
+1

*0
+1
 
²
 layer_regularization_losses
layer_metrics
,trainable_variables
-	variables
non_trainable_variables
metrics
.regularization_losses
layers
 
 
 
²
 layer_regularization_losses
layer_metrics
0trainable_variables
1	variables
non_trainable_variables
metrics
2regularization_losses
layers
 
 
 
²
 layer_regularization_losses
layer_metrics
4trainable_variables
5	variables
non_trainable_variables
metrics
6regularization_losses
layers
ZX
VARIABLE_VALUEle_net_fcb_n3/conv2d_2/kernel'c_add/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEle_net_fcb_n3/conv2d_2/bias%c_add/bias/.ATTRIBUTES/VARIABLE_VALUE

80
91

80
91
 
²
 layer_regularization_losses
layer_metrics
:trainable_variables
;	variables
non_trainable_variables
metrics
<regularization_losses
layers
 
 
 
²
 layer_regularization_losses
 layer_metrics
>trainable_variables
?	variables
¡non_trainable_variables
¢metrics
@regularization_losses
£layers
 
 
 
²
 ¤layer_regularization_losses
¥layer_metrics
Btrainable_variables
C	variables
¦non_trainable_variables
§metrics
Dregularization_losses
¨layers
 
 
 
²
 ©layer_regularization_losses
ªlayer_metrics
Ftrainable_variables
G	variables
«non_trainable_variables
¬metrics
Hregularization_losses
­layers
TR
VARIABLE_VALUEle_net_fcb_n3/dense/kernel$c5/kernel/.ATTRIBUTES/VARIABLE_VALUE
PN
VARIABLE_VALUEle_net_fcb_n3/dense/bias"c5/bias/.ATTRIBUTES/VARIABLE_VALUE

J0
K1

J0
K1
 
²
 ®layer_regularization_losses
¯layer_metrics
Ltrainable_variables
M	variables
°non_trainable_variables
±metrics
Nregularization_losses
²layers
 
 
 
²
 ³layer_regularization_losses
´layer_metrics
Ptrainable_variables
Q	variables
µnon_trainable_variables
¶metrics
Rregularization_losses
·layers
 
ge
VARIABLE_VALUE)le_net_fcb_n3/batch_normalization_1/gamma(affine3/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUE(le_net_fcb_n3/batch_normalization_1/beta'affine3/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE/le_net_fcb_n3/batch_normalization_1/moving_mean.affine3/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE3le_net_fcb_n3/batch_normalization_1/moving_variance2affine3/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

U0
V1

U0
V1
W2
X3
 
²
 ¸layer_regularization_losses
¹layer_metrics
Ytrainable_variables
Z	variables
ºnon_trainable_variables
»metrics
[regularization_losses
¼layers
 
ge
VARIABLE_VALUE)le_net_fcb_n3/batch_normalization_2/gamma(affine4/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUE(le_net_fcb_n3/batch_normalization_2/beta'affine4/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE/le_net_fcb_n3/batch_normalization_2/moving_mean.affine4/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE3le_net_fcb_n3/batch_normalization_2/moving_variance2affine4/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

^0
_1

^0
_1
`2
a3
 
²
 ½layer_regularization_losses
¾layer_metrics
btrainable_variables
c	variables
¿non_trainable_variables
Àmetrics
dregularization_losses
Álayers
VT
VARIABLE_VALUEle_net_fcb_n3/dense_1/kernel$f6/kernel/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUEle_net_fcb_n3/dense_1/bias"f6/bias/.ATTRIBUTES/VARIABLE_VALUE

f0
g1

f0
g1
 
²
 Âlayer_regularization_losses
Ãlayer_metrics
htrainable_variables
i	variables
Änon_trainable_variables
Åmetrics
jregularization_losses
Ælayers
`^
VARIABLE_VALUEle_net_fcb_n3/dense_2/kernel.output_layer/kernel/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUEle_net_fcb_n3/dense_2/bias,output_layer/bias/.ATTRIBUTES/VARIABLE_VALUE

l0
m1

l0
m1
 
²
 Çlayer_regularization_losses
Èlayer_metrics
ntrainable_variables
o	variables
Énon_trainable_variables
Êmetrics
pregularization_losses
Ëlayers
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
 
 
*
 0
!1
W2
X3
`4
a5

Ì0
Í1
v
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
 
 
 
 
 
 
 

 0
!1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

W0
X1
 
 
 
 

`0
a1
 
 
 
 
 
 
 
 
 
 
 
 
8

Îtotal

Ïcount
Ð	variables
Ñ	keras_api
I

Òtotal

Ócount
Ô
_fn_kwargs
Õ	variables
Ö	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

Î0
Ï1

Ð	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

Ò0
Ó1

Õ	variables
xv
VARIABLE_VALUE"Adam/le_net_fcb_n3/conv2d/kernel/m@c1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUE Adam/le_net_fcb_n3/conv2d/bias/m>c1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE.Adam/le_net_fcb_n3/batch_normalization/gamma/mDaffine1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE-Adam/le_net_fcb_n3/batch_normalization/beta/mCaffine1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE$Adam/le_net_fcb_n3/conv2d_1/kernel/m@c3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUE"Adam/le_net_fcb_n3/conv2d_1/bias/m>c3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUE$Adam/le_net_fcb_n3/conv2d_2/kernel/mCc_add/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUE"Adam/le_net_fcb_n3/conv2d_2/bias/mAc_add/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUE!Adam/le_net_fcb_n3/dense/kernel/m@c5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUEAdam/le_net_fcb_n3/dense/bias/m>c5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE0Adam/le_net_fcb_n3/batch_normalization_1/gamma/mDaffine3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE/Adam/le_net_fcb_n3/batch_normalization_1/beta/mCaffine3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE0Adam/le_net_fcb_n3/batch_normalization_2/gamma/mDaffine4/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE/Adam/le_net_fcb_n3/batch_normalization_2/beta/mCaffine4/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUE#Adam/le_net_fcb_n3/dense_1/kernel/m@f6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUE!Adam/le_net_fcb_n3/dense_1/bias/m>f6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE#Adam/le_net_fcb_n3/dense_2/kernel/mJoutput_layer/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUE!Adam/le_net_fcb_n3/dense_2/bias/mHoutput_layer/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE"Adam/le_net_fcb_n3/conv2d/kernel/v@c1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUE Adam/le_net_fcb_n3/conv2d/bias/v>c1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE.Adam/le_net_fcb_n3/batch_normalization/gamma/vDaffine1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE-Adam/le_net_fcb_n3/batch_normalization/beta/vCaffine1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE$Adam/le_net_fcb_n3/conv2d_1/kernel/v@c3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUE"Adam/le_net_fcb_n3/conv2d_1/bias/v>c3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUE$Adam/le_net_fcb_n3/conv2d_2/kernel/vCc_add/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUE"Adam/le_net_fcb_n3/conv2d_2/bias/vAc_add/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUE!Adam/le_net_fcb_n3/dense/kernel/v@c5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUEAdam/le_net_fcb_n3/dense/bias/v>c5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE0Adam/le_net_fcb_n3/batch_normalization_1/gamma/vDaffine3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE/Adam/le_net_fcb_n3/batch_normalization_1/beta/vCaffine3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE0Adam/le_net_fcb_n3/batch_normalization_2/gamma/vDaffine4/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE/Adam/le_net_fcb_n3/batch_normalization_2/beta/vCaffine4/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUE#Adam/le_net_fcb_n3/dense_1/kernel/v@f6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUE!Adam/le_net_fcb_n3/dense_1/bias/v>f6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE#Adam/le_net_fcb_n3/dense_2/kernel/vJoutput_layer/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUE!Adam/le_net_fcb_n3/dense_2/bias/vHoutput_layer/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

serving_default_input_1Placeholder*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *
dtype0*$
shape:ÿÿÿÿÿÿÿÿÿ  
ã	
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1le_net_fcb_n3/conv2d/kernelle_net_fcb_n3/conv2d/bias'le_net_fcb_n3/batch_normalization/gamma&le_net_fcb_n3/batch_normalization/beta-le_net_fcb_n3/batch_normalization/moving_mean1le_net_fcb_n3/batch_normalization/moving_variancele_net_fcb_n3/conv2d_1/kernelle_net_fcb_n3/conv2d_1/biasle_net_fcb_n3/conv2d_2/kernelle_net_fcb_n3/conv2d_2/biasle_net_fcb_n3/dense/kernelle_net_fcb_n3/dense/bias/le_net_fcb_n3/batch_normalization_1/moving_mean3le_net_fcb_n3/batch_normalization_1/moving_variance(le_net_fcb_n3/batch_normalization_1/beta)le_net_fcb_n3/batch_normalization_1/gammale_net_fcb_n3/dense_1/kernelle_net_fcb_n3/dense_1/bias/le_net_fcb_n3/batch_normalization_2/moving_mean3le_net_fcb_n3/batch_normalization_2/moving_variance(le_net_fcb_n3/batch_normalization_2/beta)le_net_fcb_n3/batch_normalization_2/gammale_net_fcb_n3/dense_2/kernelle_net_fcb_n3/dense_2/bias*$
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*:
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *+
f&R$
"__inference_signature_wrapper_5830
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Ð 
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename/le_net_fcb_n3/conv2d/kernel/Read/ReadVariableOp-le_net_fcb_n3/conv2d/bias/Read/ReadVariableOp;le_net_fcb_n3/batch_normalization/gamma/Read/ReadVariableOp:le_net_fcb_n3/batch_normalization/beta/Read/ReadVariableOpAle_net_fcb_n3/batch_normalization/moving_mean/Read/ReadVariableOpEle_net_fcb_n3/batch_normalization/moving_variance/Read/ReadVariableOp1le_net_fcb_n3/conv2d_1/kernel/Read/ReadVariableOp/le_net_fcb_n3/conv2d_1/bias/Read/ReadVariableOp1le_net_fcb_n3/conv2d_2/kernel/Read/ReadVariableOp/le_net_fcb_n3/conv2d_2/bias/Read/ReadVariableOp.le_net_fcb_n3/dense/kernel/Read/ReadVariableOp,le_net_fcb_n3/dense/bias/Read/ReadVariableOp=le_net_fcb_n3/batch_normalization_1/gamma/Read/ReadVariableOp<le_net_fcb_n3/batch_normalization_1/beta/Read/ReadVariableOpCle_net_fcb_n3/batch_normalization_1/moving_mean/Read/ReadVariableOpGle_net_fcb_n3/batch_normalization_1/moving_variance/Read/ReadVariableOp=le_net_fcb_n3/batch_normalization_2/gamma/Read/ReadVariableOp<le_net_fcb_n3/batch_normalization_2/beta/Read/ReadVariableOpCle_net_fcb_n3/batch_normalization_2/moving_mean/Read/ReadVariableOpGle_net_fcb_n3/batch_normalization_2/moving_variance/Read/ReadVariableOp0le_net_fcb_n3/dense_1/kernel/Read/ReadVariableOp.le_net_fcb_n3/dense_1/bias/Read/ReadVariableOp0le_net_fcb_n3/dense_2/kernel/Read/ReadVariableOp.le_net_fcb_n3/dense_2/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp6Adam/le_net_fcb_n3/conv2d/kernel/m/Read/ReadVariableOp4Adam/le_net_fcb_n3/conv2d/bias/m/Read/ReadVariableOpBAdam/le_net_fcb_n3/batch_normalization/gamma/m/Read/ReadVariableOpAAdam/le_net_fcb_n3/batch_normalization/beta/m/Read/ReadVariableOp8Adam/le_net_fcb_n3/conv2d_1/kernel/m/Read/ReadVariableOp6Adam/le_net_fcb_n3/conv2d_1/bias/m/Read/ReadVariableOp8Adam/le_net_fcb_n3/conv2d_2/kernel/m/Read/ReadVariableOp6Adam/le_net_fcb_n3/conv2d_2/bias/m/Read/ReadVariableOp5Adam/le_net_fcb_n3/dense/kernel/m/Read/ReadVariableOp3Adam/le_net_fcb_n3/dense/bias/m/Read/ReadVariableOpDAdam/le_net_fcb_n3/batch_normalization_1/gamma/m/Read/ReadVariableOpCAdam/le_net_fcb_n3/batch_normalization_1/beta/m/Read/ReadVariableOpDAdam/le_net_fcb_n3/batch_normalization_2/gamma/m/Read/ReadVariableOpCAdam/le_net_fcb_n3/batch_normalization_2/beta/m/Read/ReadVariableOp7Adam/le_net_fcb_n3/dense_1/kernel/m/Read/ReadVariableOp5Adam/le_net_fcb_n3/dense_1/bias/m/Read/ReadVariableOp7Adam/le_net_fcb_n3/dense_2/kernel/m/Read/ReadVariableOp5Adam/le_net_fcb_n3/dense_2/bias/m/Read/ReadVariableOp6Adam/le_net_fcb_n3/conv2d/kernel/v/Read/ReadVariableOp4Adam/le_net_fcb_n3/conv2d/bias/v/Read/ReadVariableOpBAdam/le_net_fcb_n3/batch_normalization/gamma/v/Read/ReadVariableOpAAdam/le_net_fcb_n3/batch_normalization/beta/v/Read/ReadVariableOp8Adam/le_net_fcb_n3/conv2d_1/kernel/v/Read/ReadVariableOp6Adam/le_net_fcb_n3/conv2d_1/bias/v/Read/ReadVariableOp8Adam/le_net_fcb_n3/conv2d_2/kernel/v/Read/ReadVariableOp6Adam/le_net_fcb_n3/conv2d_2/bias/v/Read/ReadVariableOp5Adam/le_net_fcb_n3/dense/kernel/v/Read/ReadVariableOp3Adam/le_net_fcb_n3/dense/bias/v/Read/ReadVariableOpDAdam/le_net_fcb_n3/batch_normalization_1/gamma/v/Read/ReadVariableOpCAdam/le_net_fcb_n3/batch_normalization_1/beta/v/Read/ReadVariableOpDAdam/le_net_fcb_n3/batch_normalization_2/gamma/v/Read/ReadVariableOpCAdam/le_net_fcb_n3/batch_normalization_2/beta/v/Read/ReadVariableOp7Adam/le_net_fcb_n3/dense_1/kernel/v/Read/ReadVariableOp5Adam/le_net_fcb_n3/dense_1/bias/v/Read/ReadVariableOp7Adam/le_net_fcb_n3/dense_2/kernel/v/Read/ReadVariableOp5Adam/le_net_fcb_n3/dense_2/bias/v/Read/ReadVariableOpConst*R
TinK
I2G	*
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
GPU 2J 8 *&
f!R
__inference__traced_save_7053
ç
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamele_net_fcb_n3/conv2d/kernelle_net_fcb_n3/conv2d/bias'le_net_fcb_n3/batch_normalization/gamma&le_net_fcb_n3/batch_normalization/beta-le_net_fcb_n3/batch_normalization/moving_mean1le_net_fcb_n3/batch_normalization/moving_variancele_net_fcb_n3/conv2d_1/kernelle_net_fcb_n3/conv2d_1/biasle_net_fcb_n3/conv2d_2/kernelle_net_fcb_n3/conv2d_2/biasle_net_fcb_n3/dense/kernelle_net_fcb_n3/dense/bias)le_net_fcb_n3/batch_normalization_1/gamma(le_net_fcb_n3/batch_normalization_1/beta/le_net_fcb_n3/batch_normalization_1/moving_mean3le_net_fcb_n3/batch_normalization_1/moving_variance)le_net_fcb_n3/batch_normalization_2/gamma(le_net_fcb_n3/batch_normalization_2/beta/le_net_fcb_n3/batch_normalization_2/moving_mean3le_net_fcb_n3/batch_normalization_2/moving_variancele_net_fcb_n3/dense_1/kernelle_net_fcb_n3/dense_1/biasle_net_fcb_n3/dense_2/kernelle_net_fcb_n3/dense_2/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1"Adam/le_net_fcb_n3/conv2d/kernel/m Adam/le_net_fcb_n3/conv2d/bias/m.Adam/le_net_fcb_n3/batch_normalization/gamma/m-Adam/le_net_fcb_n3/batch_normalization/beta/m$Adam/le_net_fcb_n3/conv2d_1/kernel/m"Adam/le_net_fcb_n3/conv2d_1/bias/m$Adam/le_net_fcb_n3/conv2d_2/kernel/m"Adam/le_net_fcb_n3/conv2d_2/bias/m!Adam/le_net_fcb_n3/dense/kernel/mAdam/le_net_fcb_n3/dense/bias/m0Adam/le_net_fcb_n3/batch_normalization_1/gamma/m/Adam/le_net_fcb_n3/batch_normalization_1/beta/m0Adam/le_net_fcb_n3/batch_normalization_2/gamma/m/Adam/le_net_fcb_n3/batch_normalization_2/beta/m#Adam/le_net_fcb_n3/dense_1/kernel/m!Adam/le_net_fcb_n3/dense_1/bias/m#Adam/le_net_fcb_n3/dense_2/kernel/m!Adam/le_net_fcb_n3/dense_2/bias/m"Adam/le_net_fcb_n3/conv2d/kernel/v Adam/le_net_fcb_n3/conv2d/bias/v.Adam/le_net_fcb_n3/batch_normalization/gamma/v-Adam/le_net_fcb_n3/batch_normalization/beta/v$Adam/le_net_fcb_n3/conv2d_1/kernel/v"Adam/le_net_fcb_n3/conv2d_1/bias/v$Adam/le_net_fcb_n3/conv2d_2/kernel/v"Adam/le_net_fcb_n3/conv2d_2/bias/v!Adam/le_net_fcb_n3/dense/kernel/vAdam/le_net_fcb_n3/dense/bias/v0Adam/le_net_fcb_n3/batch_normalization_1/gamma/v/Adam/le_net_fcb_n3/batch_normalization_1/beta/v0Adam/le_net_fcb_n3/batch_normalization_2/gamma/v/Adam/le_net_fcb_n3/batch_normalization_2/beta/v#Adam/le_net_fcb_n3/dense_1/kernel/v!Adam/le_net_fcb_n3/dense_1/bias/v#Adam/le_net_fcb_n3/dense_2/kernel/v!Adam/le_net_fcb_n3/dense_2/bias/v*Q
TinJ
H2F*
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
GPU 2J 8 *)
f$R"
 __inference__traced_restore_7270â
	
ª
B__inference_conv2d_2_layer_call_and_return_conditional_losses_5260

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOp¥
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ@:::W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Ñ
Ù
,__inference_le_net_fcb_n3_layer_call_fn_5714
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22
identity¢StatefulPartitionedCall£
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22*$
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*:
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_le_net_fcb_n3_layer_call_and_return_conditional_losses_56632
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*
_input_shapes}
{:ÿÿÿÿÿÿÿÿÿ  ::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
!
_user_specified_name	input_1

ü
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_4926

inputs 
cast_readvariableop_resource"
cast_1_readvariableop_resource"
cast_2_readvariableop_resource"
cast_3_readvariableop_resource
identity
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes	
:ð*
dtype02
Cast/ReadVariableOp
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes	
:ð*
dtype02
Cast_1/ReadVariableOp
Cast_2/ReadVariableOpReadVariableOpcast_2_readvariableop_resource*
_output_shapes	
:ð*
dtype02
Cast_2/ReadVariableOp
Cast_3/ReadVariableOpReadVariableOpcast_3_readvariableop_resource*
_output_shapes	
:ð*
dtype02
Cast_3/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2Cast_1/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:ð2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:ð2
batchnorm/Rsqrt
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:ð2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿð2
batchnorm/mul_1
batchnorm/mul_2MulCast/ReadVariableOp:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:ð2
batchnorm/mul_2
batchnorm/subSubCast_2/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:ð2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿð2
batchnorm/add_1h
IdentityIdentitybatchnorm/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿð2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿð:::::P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿð
 
_user_specified_nameinputs
Å
ª
M__inference_batch_normalization_layer_call_and_return_conditional_losses_6325

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ê
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3ÿ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1¦
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
î
i
K__inference_spatial_dropout2d_layer_call_and_return_conditional_losses_6417

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity_1"!

identity_1Identity_1:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
©
J
.__inference_max_pooling2d_2_layer_call_fn_4797

inputs
identityê
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_47912
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

ü
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_6675

inputs 
cast_readvariableop_resource"
cast_1_readvariableop_resource"
cast_2_readvariableop_resource"
cast_3_readvariableop_resource
identity
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes	
:ð*
dtype02
Cast/ReadVariableOp
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes	
:ð*
dtype02
Cast_1/ReadVariableOp
Cast_2/ReadVariableOpReadVariableOpcast_2_readvariableop_resource*
_output_shapes	
:ð*
dtype02
Cast_2/ReadVariableOp
Cast_3/ReadVariableOpReadVariableOpcast_3_readvariableop_resource*
_output_shapes	
:ð*
dtype02
Cast_3/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2Cast_1/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:ð2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:ð2
batchnorm/Rsqrt
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:ð2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿð2
batchnorm/mul_1
batchnorm/mul_2MulCast/ReadVariableOp:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:ð2
batchnorm/mul_2
batchnorm/subSubCast_2/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:ð2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿð2
batchnorm/add_1h
IdentityIdentitybatchnorm/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿð2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿð:::::P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿð
 
_user_specified_nameinputs
¸
i
0__inference_spatial_dropout2d_layer_call_fn_6460

inputs
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_spatial_dropout2d_layer_call_and_return_conditional_losses_47042
StatefulPartitionedCall±
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ22
StatefulPartitionedCallStatefulPartitionedCall:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
©
©
A__inference_dense_1_layer_call_and_return_conditional_losses_6794

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	ðT*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:T*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿð:::P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿð
 
_user_specified_nameinputs

`
A__inference_dropout_layer_call_and_return_conditional_losses_6604

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿð2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shapeµ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿð*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2
dropout/GreaterEqual/y¿
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿð2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿð2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿð2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿð2

Identity"
identityIdentity:output:0*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿð:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿð
 
_user_specified_nameinputs
Ý
k
M__inference_spatial_dropout2d_1_layer_call_and_return_conditional_losses_4782

inputs

identity_1}
IdentityIdentityinputs*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity

Identity_1IdentityIdentity:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity_1"!

identity_1Identity_1:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ø
|
'__inference_conv2d_1_layer_call_fn_6389

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallú
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_conv2d_1_layer_call_and_return_conditional_losses_51942
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ ::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
ÆM

G__inference_le_net_fcb_n3_layer_call_and_return_conditional_losses_5663

inputs
conv2d_5594
conv2d_5596
batch_normalization_5599
batch_normalization_5601
batch_normalization_5603
batch_normalization_5605
conv2d_1_5610
conv2d_1_5612
conv2d_2_5616
conv2d_2_5618

dense_5626

dense_5628
batch_normalization_1_5632
batch_normalization_1_5634
batch_normalization_1_5636
batch_normalization_1_5638
dense_1_5642
dense_1_5644
batch_normalization_2_5647
batch_normalization_2_5649
batch_normalization_2_5651
batch_normalization_2_5653
dense_2_5657
dense_2_5659
identity¢+batch_normalization/StatefulPartitionedCall¢-batch_normalization_1/StatefulPartitionedCall¢-batch_normalization_2/StatefulPartitionedCall¢conv2d/StatefulPartitionedCall¢ conv2d_1/StatefulPartitionedCall¢ conv2d_2/StatefulPartitionedCall¢dense/StatefulPartitionedCall¢dense_1/StatefulPartitionedCall¢dense_2/StatefulPartitionedCall
conv2d/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_5594conv2d_5596*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_conv2d_layer_call_and_return_conditional_losses_50922 
conv2d/StatefulPartitionedCall¦
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0batch_normalization_5599batch_normalization_5601batch_normalization_5603batch_normalization_5605*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_batch_normalization_layer_call_and_return_conditional_losses_51452-
+batch_normalization/StatefulPartitionedCall
ReluRelu4batch_normalization/StatefulPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Reluõ
max_pooling2d/PartitionedCallPartitionedCallRelu:activations:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_46312
max_pooling2d/PartitionedCall¶
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0conv2d_1_5610conv2d_1_5612*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_conv2d_1_layer_call_and_return_conditional_losses_51942"
 conv2d_1/StatefulPartitionedCall
!spatial_dropout2d/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_spatial_dropout2d_layer_call_and_return_conditional_losses_52372#
!spatial_dropout2d/PartitionedCall»
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall*spatial_dropout2d/PartitionedCall:output:0conv2d_2_5616conv2d_2_5618*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_conv2d_2_layer_call_and_return_conditional_losses_52602"
 conv2d_2/StatefulPartitionedCall
#spatial_dropout2d_1/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_spatial_dropout2d_1_layer_call_and_return_conditional_losses_53032%
#spatial_dropout2d_1/PartitionedCall
max_pooling2d_2/PartitionedCallPartitionedCall,spatial_dropout2d_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_47912!
max_pooling2d_2/PartitionedCall}
Relu_1Relu(max_pooling2d_2/PartitionedCall:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu_1þ
max_pooling2d_1/PartitionedCallPartitionedCallRelu_1:activations:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_46432!
max_pooling2d_1/PartitionedCallò
flatten/PartitionedCallPartitionedCall(max_pooling2d_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_flatten_layer_call_and_return_conditional_losses_53242
flatten/PartitionedCall
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0
dense_5626
dense_5628*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿð*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_53432
dense/StatefulPartitionedCallð
dropout/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿð* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_53762
dropout/PartitionedCall¦
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0batch_normalization_1_5632batch_normalization_1_5634batch_normalization_1_5636batch_normalization_1_5638*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿð*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_49262/
-batch_normalization_1/StatefulPartitionedCall
Relu_2Relu6batch_normalization_1/StatefulPartitionedCall:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿð2
Relu_2
dense_1/StatefulPartitionedCallStatefulPartitionedCallRelu_2:activations:0dense_1_5642dense_1_5644*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_54362!
dense_1/StatefulPartitionedCall­
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0batch_normalization_2_5647batch_normalization_2_5649batch_normalization_2_5651batch_normalization_2_5653*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_50662/
-batch_normalization_2/StatefulPartitionedCall
SoftmaxSoftmax6batch_normalization_2/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT2	
Softmax
dense_2/StatefulPartitionedCallStatefulPartitionedCallSoftmax:softmax:0dense_2_5657dense_2_5659*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dense_2_layer_call_and_return_conditional_losses_54992!
dense_2/StatefulPartitionedCallÕ
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*
_input_shapes}
{:ÿÿÿÿÿÿÿÿÿ  ::::::::::::::::::::::::2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
 
_user_specified_nameinputs
Ï
¥
2__inference_batch_normalization_layer_call_fn_6305

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_batch_normalization_layer_call_and_return_conditional_losses_51452
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ ::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs

j
K__inference_spatial_dropout2d_layer_call_and_return_conditional_losses_6412

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ì
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout/Mul
dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2 
dropout/random_uniform/shape/1
dropout/random_uniform/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2 
dropout/random_uniform/shape/2ö
dropout/random_uniform/shapePackstrided_slice:output:0'dropout/random_uniform/shape/1:output:0'dropout/random_uniform/shape/2:output:0strided_slice_1:output:0*
N*
T0*
_output_shapes
:2
dropout/random_uniform/shapeÔ
$dropout/random_uniform/RandomUniformRandomUniform%dropout/random_uniform/shape:output:0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2
dropout/GreaterEqual/yÏ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Ö
{
&__inference_dense_2_layer_call_fn_6823

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallñ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dense_2_layer_call_and_return_conditional_losses_54992
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿT::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT
 
_user_specified_nameinputs


M__inference_batch_normalization_layer_call_and_return_conditional_losses_6343

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ü
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ :::::i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
ý
ª
M__inference_batch_normalization_layer_call_and_return_conditional_losses_6261

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ø
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3ÿ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs

_
&__inference_dropout_layer_call_fn_6614

inputs
identity¢StatefulPartitionedCallØ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿð* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_53712
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿð2

Identity"
identityIdentity:output:0*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿð22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿð
 
_user_specified_nameinputs
Û(
¸
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_6655

inputs
assignmovingavg_6630
assignmovingavg_1_6636 
cast_readvariableop_resource"
cast_1_readvariableop_resource
identity¢#AssignMovingAvg/AssignSubVariableOp¢%AssignMovingAvg_1/AssignSubVariableOp
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indices
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	ð*
	keep_dims(2
moments/mean}
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	ð2
moments/StopGradient¥
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿð2
moments/SquaredDifference
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices³
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	ð*
	keep_dims(2
moments/variance
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:ð*
squeeze_dims
 2
moments/Squeeze
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:ð*
squeeze_dims
 2
moments/Squeeze_1
AssignMovingAvg/decayConst*'
_class
loc:@AssignMovingAvg/6630*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg/decay
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_6630*
_output_shapes	
:ð*
dtype02 
AssignMovingAvg/ReadVariableOpÂ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*'
_class
loc:@AssignMovingAvg/6630*
_output_shapes	
:ð2
AssignMovingAvg/sub¹
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*'
_class
loc:@AssignMovingAvg/6630*
_output_shapes	
:ð2
AssignMovingAvg/mulý
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_6630AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*'
_class
loc:@AssignMovingAvg/6630*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp¢
AssignMovingAvg_1/decayConst*)
_class
loc:@AssignMovingAvg_1/6636*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg_1/decay
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_6636*
_output_shapes	
:ð*
dtype02"
 AssignMovingAvg_1/ReadVariableOpÌ
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*)
_class
loc:@AssignMovingAvg_1/6636*
_output_shapes	
:ð2
AssignMovingAvg_1/subÃ
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*)
_class
loc:@AssignMovingAvg_1/6636*
_output_shapes	
:ð2
AssignMovingAvg_1/mul
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_6636AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*)
_class
loc:@AssignMovingAvg_1/6636*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes	
:ð*
dtype02
Cast/ReadVariableOp
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes	
:ð*
dtype02
Cast_1/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:ð2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:ð2
batchnorm/Rsqrt
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes	
:ð2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿð2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:ð2
batchnorm/mul_2}
batchnorm/subSubCast/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:ð2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿð2
batchnorm/add_1¶
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿð2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿð::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿð
 
_user_specified_nameinputs
·
§
4__inference_batch_normalization_1_layer_call_fn_6701

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿð*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_49262
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿð2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿð::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿð
 
_user_specified_nameinputs
®
©
A__inference_dense_2_layer_call_and_return_conditional_losses_5499

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:T*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Softmaxe
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿT:::O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT
 
_user_specified_nameinputs
Ì

M__inference_batch_normalization_layer_call_and_return_conditional_losses_6279

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ê
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
is_training( 2
FusedBatchNormV3p
IdentityIdentityFusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ :::::W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs

`
A__inference_dropout_layer_call_and_return_conditional_losses_5371

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿð2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shapeµ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿð*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2
dropout/GreaterEqual/y¿
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿð2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿð2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿð2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿð2

Identity"
identityIdentity:output:0*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿð:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿð
 
_user_specified_nameinputs
ú
|
'__inference_conv2d_2_layer_call_fn_6485

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallû
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_conv2d_2_layer_call_and_return_conditional_losses_52602
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ@::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
©
©
A__inference_dense_1_layer_call_and_return_conditional_losses_5436

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	ðT*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:T*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿð:::P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿð
 
_user_specified_nameinputs
ÉM

G__inference_le_net_fcb_n3_layer_call_and_return_conditional_losses_5588
input_1
conv2d_5519
conv2d_5521
batch_normalization_5524
batch_normalization_5526
batch_normalization_5528
batch_normalization_5530
conv2d_1_5535
conv2d_1_5537
conv2d_2_5541
conv2d_2_5543

dense_5551

dense_5553
batch_normalization_1_5557
batch_normalization_1_5559
batch_normalization_1_5561
batch_normalization_1_5563
dense_1_5567
dense_1_5569
batch_normalization_2_5572
batch_normalization_2_5574
batch_normalization_2_5576
batch_normalization_2_5578
dense_2_5582
dense_2_5584
identity¢+batch_normalization/StatefulPartitionedCall¢-batch_normalization_1/StatefulPartitionedCall¢-batch_normalization_2/StatefulPartitionedCall¢conv2d/StatefulPartitionedCall¢ conv2d_1/StatefulPartitionedCall¢ conv2d_2/StatefulPartitionedCall¢dense/StatefulPartitionedCall¢dense_1/StatefulPartitionedCall¢dense_2/StatefulPartitionedCall
conv2d/StatefulPartitionedCallStatefulPartitionedCallinput_1conv2d_5519conv2d_5521*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_conv2d_layer_call_and_return_conditional_losses_50922 
conv2d/StatefulPartitionedCall¦
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0batch_normalization_5524batch_normalization_5526batch_normalization_5528batch_normalization_5530*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_batch_normalization_layer_call_and_return_conditional_losses_51452-
+batch_normalization/StatefulPartitionedCall
ReluRelu4batch_normalization/StatefulPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Reluõ
max_pooling2d/PartitionedCallPartitionedCallRelu:activations:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_46312
max_pooling2d/PartitionedCall¶
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0conv2d_1_5535conv2d_1_5537*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_conv2d_1_layer_call_and_return_conditional_losses_51942"
 conv2d_1/StatefulPartitionedCall
!spatial_dropout2d/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_spatial_dropout2d_layer_call_and_return_conditional_losses_52372#
!spatial_dropout2d/PartitionedCall»
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall*spatial_dropout2d/PartitionedCall:output:0conv2d_2_5541conv2d_2_5543*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_conv2d_2_layer_call_and_return_conditional_losses_52602"
 conv2d_2/StatefulPartitionedCall
#spatial_dropout2d_1/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_spatial_dropout2d_1_layer_call_and_return_conditional_losses_53032%
#spatial_dropout2d_1/PartitionedCall
max_pooling2d_2/PartitionedCallPartitionedCall,spatial_dropout2d_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_47912!
max_pooling2d_2/PartitionedCall}
Relu_1Relu(max_pooling2d_2/PartitionedCall:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu_1þ
max_pooling2d_1/PartitionedCallPartitionedCallRelu_1:activations:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_46432!
max_pooling2d_1/PartitionedCallò
flatten/PartitionedCallPartitionedCall(max_pooling2d_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_flatten_layer_call_and_return_conditional_losses_53242
flatten/PartitionedCall
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0
dense_5551
dense_5553*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿð*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_53432
dense/StatefulPartitionedCallð
dropout/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿð* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_53762
dropout/PartitionedCall¦
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0batch_normalization_1_5557batch_normalization_1_5559batch_normalization_1_5561batch_normalization_1_5563*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿð*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_49262/
-batch_normalization_1/StatefulPartitionedCall
Relu_2Relu6batch_normalization_1/StatefulPartitionedCall:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿð2
Relu_2
dense_1/StatefulPartitionedCallStatefulPartitionedCallRelu_2:activations:0dense_1_5567dense_1_5569*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_54362!
dense_1/StatefulPartitionedCall­
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0batch_normalization_2_5572batch_normalization_2_5574batch_normalization_2_5576batch_normalization_2_5578*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_50662/
-batch_normalization_2/StatefulPartitionedCall
SoftmaxSoftmax6batch_normalization_2/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT2	
Softmax
dense_2/StatefulPartitionedCallStatefulPartitionedCallSoftmax:softmax:0dense_2_5582dense_2_5584*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dense_2_layer_call_and_return_conditional_losses_54992!
dense_2/StatefulPartitionedCallÕ
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*
_input_shapes}
{:ÿÿÿÿÿÿÿÿÿ  ::::::::::::::::::::::::2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall:X T
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
!
_user_specified_name	input_1
­
§
?__inference_dense_layer_call_and_return_conditional_losses_6583

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
ð*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿð2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ð*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿð2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿð2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿð2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:::P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

ü
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_6757

inputs 
cast_readvariableop_resource"
cast_1_readvariableop_resource"
cast_2_readvariableop_resource"
cast_3_readvariableop_resource
identity
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:T*
dtype02
Cast/ReadVariableOp
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
:T*
dtype02
Cast_1/ReadVariableOp
Cast_2/ReadVariableOpReadVariableOpcast_2_readvariableop_resource*
_output_shapes
:T*
dtype02
Cast_2/ReadVariableOp
Cast_3/ReadVariableOpReadVariableOpcast_3_readvariableop_resource*
_output_shapes
:T*
dtype02
Cast_3/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2Cast_1/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:T2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:T2
batchnorm/Rsqrt~
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:T2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT2
batchnorm/mul_1~
batchnorm/mul_2MulCast/ReadVariableOp:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:T2
batchnorm/mul_2~
batchnorm/subSubCast_2/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:T2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT2
batchnorm/add_1g
IdentityIdentitybatchnorm/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿT:::::O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT
 
_user_specified_nameinputs
Ø
{
&__inference_dense_1_layer_call_fn_6803

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallñ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_54362
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿð::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿð
 
_user_specified_nameinputs
Ì

M__inference_batch_normalization_layer_call_and_return_conditional_losses_5145

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ê
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
is_training( 2
FusedBatchNormV3p
IdentityIdentityFusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ :::::W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
¼
k
2__inference_spatial_dropout2d_1_layer_call_fn_6518

inputs
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_spatial_dropout2d_1_layer_call_and_return_conditional_losses_47722
StatefulPartitionedCall±
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ22
StatefulPartitionedCallStatefulPartitionedCall:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ö
y
$__inference_dense_layer_call_fn_6592

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallð
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿð*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_53432
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿð2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
µ
§
4__inference_batch_normalization_1_layer_call_fn_6688

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿð*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_48932
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿð2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿð::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿð
 
_user_specified_nameinputs

j
K__inference_spatial_dropout2d_layer_call_and_return_conditional_losses_5232

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ì
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout/Mul
dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2 
dropout/random_uniform/shape/1
dropout/random_uniform/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2 
dropout/random_uniform/shape/2ö
dropout/random_uniform/shapePackstrided_slice:output:0'dropout/random_uniform/shape/1:output:0'dropout/random_uniform/shape/2:output:0strided_slice_1:output:0*
N*
T0*
_output_shapes
:2
dropout/random_uniform/shapeÔ
$dropout/random_uniform/RandomUniformRandomUniform%dropout/random_uniform/shape:output:0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2
dropout/GreaterEqual/yÏ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs

¥
2__inference_batch_normalization_layer_call_fn_6369

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall±
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_batch_normalization_layer_call_and_return_conditional_losses_46142
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
¥
H
,__inference_max_pooling2d_layer_call_fn_4637

inputs
identityè
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_46312
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Û
i
K__inference_spatial_dropout2d_layer_call_and_return_conditional_losses_6455

inputs

identity_1}
IdentityIdentityinputs*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity

Identity_1IdentityIdentity:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity_1"!

identity_1Identity_1:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

ü
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_5066

inputs 
cast_readvariableop_resource"
cast_1_readvariableop_resource"
cast_2_readvariableop_resource"
cast_3_readvariableop_resource
identity
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:T*
dtype02
Cast/ReadVariableOp
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
:T*
dtype02
Cast_1/ReadVariableOp
Cast_2/ReadVariableOpReadVariableOpcast_2_readvariableop_resource*
_output_shapes
:T*
dtype02
Cast_2/ReadVariableOp
Cast_3/ReadVariableOpReadVariableOpcast_3_readvariableop_resource*
_output_shapes
:T*
dtype02
Cast_3/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2Cast_1/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:T2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:T2
batchnorm/Rsqrt~
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:T2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT2
batchnorm/mul_1~
batchnorm/mul_2MulCast/ReadVariableOp:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:T2
batchnorm/mul_2~
batchnorm/subSubCast_2/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:T2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT2
batchnorm/add_1g
IdentityIdentitybatchnorm/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿT:::::O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT
 
_user_specified_nameinputs
È
_
A__inference_dropout_layer_call_and_return_conditional_losses_6609

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿð2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿð2

Identity_1"!

identity_1Identity_1:output:0*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿð:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿð
 
_user_specified_nameinputs
±
N
2__inference_spatial_dropout2d_1_layer_call_fn_6523

inputs
identityî
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_spatial_dropout2d_1_layer_call_and_return_conditional_losses_47822
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ý
k
M__inference_spatial_dropout2d_1_layer_call_and_return_conditional_losses_6513

inputs

identity_1}
IdentityIdentityinputs*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity

Identity_1IdentityIdentity:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity_1"!

identity_1Identity_1:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

Ì
G__inference_le_net_fcb_n3_layer_call_and_return_conditional_losses_6011

inputs)
%conv2d_conv2d_readvariableop_resource*
&conv2d_biasadd_readvariableop_resource/
+batch_normalization_readvariableop_resource1
-batch_normalization_readvariableop_1_resource@
<batch_normalization_fusedbatchnormv3_readvariableop_resourceB
>batch_normalization_fusedbatchnormv3_readvariableop_1_resource+
'conv2d_1_conv2d_readvariableop_resource,
(conv2d_1_biasadd_readvariableop_resource+
'conv2d_2_conv2d_readvariableop_resource,
(conv2d_2_biasadd_readvariableop_resource(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource.
*batch_normalization_1_assignmovingavg_59380
,batch_normalization_1_assignmovingavg_1_59446
2batch_normalization_1_cast_readvariableop_resource8
4batch_normalization_1_cast_1_readvariableop_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource.
*batch_normalization_2_assignmovingavg_59780
,batch_normalization_2_assignmovingavg_1_59846
2batch_normalization_2_cast_readvariableop_resource8
4batch_normalization_2_cast_1_readvariableop_resource*
&dense_2_matmul_readvariableop_resource+
'dense_2_biasadd_readvariableop_resource
identity¢"batch_normalization/AssignNewValue¢$batch_normalization/AssignNewValue_1¢9batch_normalization_1/AssignMovingAvg/AssignSubVariableOp¢;batch_normalization_1/AssignMovingAvg_1/AssignSubVariableOp¢9batch_normalization_2/AssignMovingAvg/AssignSubVariableOp¢;batch_normalization_2/AssignMovingAvg_1/AssignSubVariableOpª
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
conv2d/Conv2D/ReadVariableOp¹
conv2d/Conv2DConv2Dinputs$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingVALID*
strides
2
conv2d/Conv2D¡
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
conv2d/BiasAdd/ReadVariableOp¤
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
conv2d/BiasAddu
conv2d/ReluReluconv2d/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
conv2d/Relu°
"batch_normalization/ReadVariableOpReadVariableOp+batch_normalization_readvariableop_resource*
_output_shapes
: *
dtype02$
"batch_normalization/ReadVariableOp¶
$batch_normalization/ReadVariableOp_1ReadVariableOp-batch_normalization_readvariableop_1_resource*
_output_shapes
: *
dtype02&
$batch_normalization/ReadVariableOp_1ã
3batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype025
3batch_normalization/FusedBatchNormV3/ReadVariableOpé
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype027
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1ã
$batch_normalization/FusedBatchNormV3FusedBatchNormV3conv2d/Relu:activations:0*batch_normalization/ReadVariableOp:value:0,batch_normalization/ReadVariableOp_1:value:0;batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0=batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
exponential_avg_factor%
×#<2&
$batch_normalization/FusedBatchNormV3÷
"batch_normalization/AssignNewValueAssignVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource1batch_normalization/FusedBatchNormV3:batch_mean:04^batch_normalization/FusedBatchNormV3/ReadVariableOp*O
_classE
CAloc:@batch_normalization/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02$
"batch_normalization/AssignNewValue
$batch_normalization/AssignNewValue_1AssignVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource5batch_normalization/FusedBatchNormV3:batch_variance:06^batch_normalization/FusedBatchNormV3/ReadVariableOp_1*Q
_classG
ECloc:@batch_normalization/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02&
$batch_normalization/AssignNewValue_1x
ReluRelu(batch_normalization/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Reluº
max_pooling2d/MaxPoolMaxPoolRelu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
ksize
*
paddingVALID*
strides
2
max_pooling2d/MaxPool°
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02 
conv2d_1/Conv2D/ReadVariableOp×
conv2d_1/Conv2DConv2Dmax_pooling2d/MaxPool:output:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
2
conv2d_1/Conv2D§
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_1/BiasAdd/ReadVariableOp¬
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
conv2d_1/BiasAdd{
conv2d_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
conv2d_1/Relu}
spatial_dropout2d/ShapeShapeconv2d_1/Relu:activations:0*
T0*
_output_shapes
:2
spatial_dropout2d/Shape
%spatial_dropout2d/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%spatial_dropout2d/strided_slice/stack
'spatial_dropout2d/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'spatial_dropout2d/strided_slice/stack_1
'spatial_dropout2d/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'spatial_dropout2d/strided_slice/stack_2Î
spatial_dropout2d/strided_sliceStridedSlice spatial_dropout2d/Shape:output:0.spatial_dropout2d/strided_slice/stack:output:00spatial_dropout2d/strided_slice/stack_1:output:00spatial_dropout2d/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
spatial_dropout2d/strided_slice
'spatial_dropout2d/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2)
'spatial_dropout2d/strided_slice_1/stack 
)spatial_dropout2d/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)spatial_dropout2d/strided_slice_1/stack_1 
)spatial_dropout2d/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)spatial_dropout2d/strided_slice_1/stack_2Ø
!spatial_dropout2d/strided_slice_1StridedSlice spatial_dropout2d/Shape:output:00spatial_dropout2d/strided_slice_1/stack:output:02spatial_dropout2d/strided_slice_1/stack_1:output:02spatial_dropout2d/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!spatial_dropout2d/strided_slice_1
spatial_dropout2d/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2!
spatial_dropout2d/dropout/ConstÆ
spatial_dropout2d/dropout/MulMulconv2d_1/Relu:activations:0(spatial_dropout2d/dropout/Const:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
spatial_dropout2d/dropout/Mul¦
0spatial_dropout2d/dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :22
0spatial_dropout2d/dropout/random_uniform/shape/1¦
0spatial_dropout2d/dropout/random_uniform/shape/2Const*
_output_shapes
: *
dtype0*
value	B :22
0spatial_dropout2d/dropout/random_uniform/shape/2â
.spatial_dropout2d/dropout/random_uniform/shapePack(spatial_dropout2d/strided_slice:output:09spatial_dropout2d/dropout/random_uniform/shape/1:output:09spatial_dropout2d/dropout/random_uniform/shape/2:output:0*spatial_dropout2d/strided_slice_1:output:0*
N*
T0*
_output_shapes
:20
.spatial_dropout2d/dropout/random_uniform/shape
6spatial_dropout2d/dropout/random_uniform/RandomUniformRandomUniform7spatial_dropout2d/dropout/random_uniform/shape:output:0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
dtype028
6spatial_dropout2d/dropout/random_uniform/RandomUniform
(spatial_dropout2d/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2*
(spatial_dropout2d/dropout/GreaterEqual/y
&spatial_dropout2d/dropout/GreaterEqualGreaterEqual?spatial_dropout2d/dropout/random_uniform/RandomUniform:output:01spatial_dropout2d/dropout/GreaterEqual/y:output:0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2(
&spatial_dropout2d/dropout/GreaterEqualÆ
spatial_dropout2d/dropout/CastCast*spatial_dropout2d/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2 
spatial_dropout2d/dropout/CastÊ
spatial_dropout2d/dropout/Mul_1Mul!spatial_dropout2d/dropout/Mul:z:0"spatial_dropout2d/dropout/Cast:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2!
spatial_dropout2d/dropout/Mul_1±
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02 
conv2d_2/Conv2D/ReadVariableOpÝ
conv2d_2/Conv2DConv2D#spatial_dropout2d/dropout/Mul_1:z:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2
conv2d_2/Conv2D¨
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02!
conv2d_2/BiasAdd/ReadVariableOp­
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv2d_2/BiasAdd|
conv2d_2/ReluReluconv2d_2/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv2d_2/Relu
spatial_dropout2d_1/ShapeShapeconv2d_2/Relu:activations:0*
T0*
_output_shapes
:2
spatial_dropout2d_1/Shape
'spatial_dropout2d_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'spatial_dropout2d_1/strided_slice/stack 
)spatial_dropout2d_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)spatial_dropout2d_1/strided_slice/stack_1 
)spatial_dropout2d_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)spatial_dropout2d_1/strided_slice/stack_2Ú
!spatial_dropout2d_1/strided_sliceStridedSlice"spatial_dropout2d_1/Shape:output:00spatial_dropout2d_1/strided_slice/stack:output:02spatial_dropout2d_1/strided_slice/stack_1:output:02spatial_dropout2d_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!spatial_dropout2d_1/strided_slice 
)spatial_dropout2d_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2+
)spatial_dropout2d_1/strided_slice_1/stack¤
+spatial_dropout2d_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+spatial_dropout2d_1/strided_slice_1/stack_1¤
+spatial_dropout2d_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+spatial_dropout2d_1/strided_slice_1/stack_2ä
#spatial_dropout2d_1/strided_slice_1StridedSlice"spatial_dropout2d_1/Shape:output:02spatial_dropout2d_1/strided_slice_1/stack:output:04spatial_dropout2d_1/strided_slice_1/stack_1:output:04spatial_dropout2d_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#spatial_dropout2d_1/strided_slice_1
!spatial_dropout2d_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2#
!spatial_dropout2d_1/dropout/ConstÍ
spatial_dropout2d_1/dropout/MulMulconv2d_2/Relu:activations:0*spatial_dropout2d_1/dropout/Const:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
spatial_dropout2d_1/dropout/Mulª
2spatial_dropout2d_1/dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :24
2spatial_dropout2d_1/dropout/random_uniform/shape/1ª
2spatial_dropout2d_1/dropout/random_uniform/shape/2Const*
_output_shapes
: *
dtype0*
value	B :24
2spatial_dropout2d_1/dropout/random_uniform/shape/2î
0spatial_dropout2d_1/dropout/random_uniform/shapePack*spatial_dropout2d_1/strided_slice:output:0;spatial_dropout2d_1/dropout/random_uniform/shape/1:output:0;spatial_dropout2d_1/dropout/random_uniform/shape/2:output:0,spatial_dropout2d_1/strided_slice_1:output:0*
N*
T0*
_output_shapes
:22
0spatial_dropout2d_1/dropout/random_uniform/shape
8spatial_dropout2d_1/dropout/random_uniform/RandomUniformRandomUniform9spatial_dropout2d_1/dropout/random_uniform/shape:output:0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
dtype02:
8spatial_dropout2d_1/dropout/random_uniform/RandomUniform
*spatial_dropout2d_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2,
*spatial_dropout2d_1/dropout/GreaterEqual/y
(spatial_dropout2d_1/dropout/GreaterEqualGreaterEqualAspatial_dropout2d_1/dropout/random_uniform/RandomUniform:output:03spatial_dropout2d_1/dropout/GreaterEqual/y:output:0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2*
(spatial_dropout2d_1/dropout/GreaterEqualÌ
 spatial_dropout2d_1/dropout/CastCast,spatial_dropout2d_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2"
 spatial_dropout2d_1/dropout/CastÓ
!spatial_dropout2d_1/dropout/Mul_1Mul#spatial_dropout2d_1/dropout/Mul:z:0$spatial_dropout2d_1/dropout/Cast:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!spatial_dropout2d_1/dropout/Mul_1Ò
max_pooling2d_2/MaxPoolMaxPool%spatial_dropout2d_1/dropout/Mul_1:z:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
2
max_pooling2d_2/MaxPoolu
Relu_1Relu max_pooling2d_2/MaxPool:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu_1Á
max_pooling2d_1/MaxPoolMaxPoolRelu_1:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
2
max_pooling2d_1/MaxPoolo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
flatten/Const
flatten/ReshapeReshape max_pooling2d_1/MaxPool:output:0flatten/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
flatten/Reshape¡
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
ð*
dtype02
dense/MatMul/ReadVariableOp
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿð2
dense/MatMul
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:ð*
dtype02
dense/BiasAdd/ReadVariableOp
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿð2
dense/BiasAddk

dense/ReluReludense/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿð2

dense/Relus
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?2
dropout/dropout/Const
dropout/dropout/MulMuldense/Relu:activations:0dropout/dropout/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿð2
dropout/dropout/Mulv
dropout/dropout/ShapeShapedense/Relu:activations:0*
T0*
_output_shapes
:2
dropout/dropout/ShapeÍ
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿð*
dtype02.
,dropout/dropout/random_uniform/RandomUniform
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2 
dropout/dropout/GreaterEqual/yß
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿð2
dropout/dropout/GreaterEqual
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿð2
dropout/dropout/Cast
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿð2
dropout/dropout/Mul_1¶
4batch_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 26
4batch_normalization_1/moments/mean/reduction_indiceså
"batch_normalization_1/moments/meanMeandropout/dropout/Mul_1:z:0=batch_normalization_1/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	ð*
	keep_dims(2$
"batch_normalization_1/moments/mean¿
*batch_normalization_1/moments/StopGradientStopGradient+batch_normalization_1/moments/mean:output:0*
T0*
_output_shapes
:	ð2,
*batch_normalization_1/moments/StopGradientú
/batch_normalization_1/moments/SquaredDifferenceSquaredDifferencedropout/dropout/Mul_1:z:03batch_normalization_1/moments/StopGradient:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿð21
/batch_normalization_1/moments/SquaredDifference¾
8batch_normalization_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2:
8batch_normalization_1/moments/variance/reduction_indices
&batch_normalization_1/moments/varianceMean3batch_normalization_1/moments/SquaredDifference:z:0Abatch_normalization_1/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	ð*
	keep_dims(2(
&batch_normalization_1/moments/varianceÃ
%batch_normalization_1/moments/SqueezeSqueeze+batch_normalization_1/moments/mean:output:0*
T0*
_output_shapes	
:ð*
squeeze_dims
 2'
%batch_normalization_1/moments/SqueezeË
'batch_normalization_1/moments/Squeeze_1Squeeze/batch_normalization_1/moments/variance:output:0*
T0*
_output_shapes	
:ð*
squeeze_dims
 2)
'batch_normalization_1/moments/Squeeze_1Þ
+batch_normalization_1/AssignMovingAvg/decayConst*=
_class3
1/loc:@batch_normalization_1/AssignMovingAvg/5938*
_output_shapes
: *
dtype0*
valueB
 *
×#<2-
+batch_normalization_1/AssignMovingAvg/decayÔ
4batch_normalization_1/AssignMovingAvg/ReadVariableOpReadVariableOp*batch_normalization_1_assignmovingavg_5938*
_output_shapes	
:ð*
dtype026
4batch_normalization_1/AssignMovingAvg/ReadVariableOp°
)batch_normalization_1/AssignMovingAvg/subSub<batch_normalization_1/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_1/moments/Squeeze:output:0*
T0*=
_class3
1/loc:@batch_normalization_1/AssignMovingAvg/5938*
_output_shapes	
:ð2+
)batch_normalization_1/AssignMovingAvg/sub§
)batch_normalization_1/AssignMovingAvg/mulMul-batch_normalization_1/AssignMovingAvg/sub:z:04batch_normalization_1/AssignMovingAvg/decay:output:0*
T0*=
_class3
1/loc:@batch_normalization_1/AssignMovingAvg/5938*
_output_shapes	
:ð2+
)batch_normalization_1/AssignMovingAvg/mul
9batch_normalization_1/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp*batch_normalization_1_assignmovingavg_5938-batch_normalization_1/AssignMovingAvg/mul:z:05^batch_normalization_1/AssignMovingAvg/ReadVariableOp*=
_class3
1/loc:@batch_normalization_1/AssignMovingAvg/5938*
_output_shapes
 *
dtype02;
9batch_normalization_1/AssignMovingAvg/AssignSubVariableOpä
-batch_normalization_1/AssignMovingAvg_1/decayConst*?
_class5
31loc:@batch_normalization_1/AssignMovingAvg_1/5944*
_output_shapes
: *
dtype0*
valueB
 *
×#<2/
-batch_normalization_1/AssignMovingAvg_1/decayÚ
6batch_normalization_1/AssignMovingAvg_1/ReadVariableOpReadVariableOp,batch_normalization_1_assignmovingavg_1_5944*
_output_shapes	
:ð*
dtype028
6batch_normalization_1/AssignMovingAvg_1/ReadVariableOpº
+batch_normalization_1/AssignMovingAvg_1/subSub>batch_normalization_1/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_1/moments/Squeeze_1:output:0*
T0*?
_class5
31loc:@batch_normalization_1/AssignMovingAvg_1/5944*
_output_shapes	
:ð2-
+batch_normalization_1/AssignMovingAvg_1/sub±
+batch_normalization_1/AssignMovingAvg_1/mulMul/batch_normalization_1/AssignMovingAvg_1/sub:z:06batch_normalization_1/AssignMovingAvg_1/decay:output:0*
T0*?
_class5
31loc:@batch_normalization_1/AssignMovingAvg_1/5944*
_output_shapes	
:ð2-
+batch_normalization_1/AssignMovingAvg_1/mul
;batch_normalization_1/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp,batch_normalization_1_assignmovingavg_1_5944/batch_normalization_1/AssignMovingAvg_1/mul:z:07^batch_normalization_1/AssignMovingAvg_1/ReadVariableOp*?
_class5
31loc:@batch_normalization_1/AssignMovingAvg_1/5944*
_output_shapes
 *
dtype02=
;batch_normalization_1/AssignMovingAvg_1/AssignSubVariableOpÆ
)batch_normalization_1/Cast/ReadVariableOpReadVariableOp2batch_normalization_1_cast_readvariableop_resource*
_output_shapes	
:ð*
dtype02+
)batch_normalization_1/Cast/ReadVariableOpÌ
+batch_normalization_1/Cast_1/ReadVariableOpReadVariableOp4batch_normalization_1_cast_1_readvariableop_resource*
_output_shapes	
:ð*
dtype02-
+batch_normalization_1/Cast_1/ReadVariableOp
%batch_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2'
%batch_normalization_1/batchnorm/add/yÛ
#batch_normalization_1/batchnorm/addAddV20batch_normalization_1/moments/Squeeze_1:output:0.batch_normalization_1/batchnorm/add/y:output:0*
T0*
_output_shapes	
:ð2%
#batch_normalization_1/batchnorm/add¦
%batch_normalization_1/batchnorm/RsqrtRsqrt'batch_normalization_1/batchnorm/add:z:0*
T0*
_output_shapes	
:ð2'
%batch_normalization_1/batchnorm/Rsqrt×
#batch_normalization_1/batchnorm/mulMul)batch_normalization_1/batchnorm/Rsqrt:y:03batch_normalization_1/Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes	
:ð2%
#batch_normalization_1/batchnorm/mulÌ
%batch_normalization_1/batchnorm/mul_1Muldropout/dropout/Mul_1:z:0'batch_normalization_1/batchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿð2'
%batch_normalization_1/batchnorm/mul_1Ô
%batch_normalization_1/batchnorm/mul_2Mul.batch_normalization_1/moments/Squeeze:output:0'batch_normalization_1/batchnorm/mul:z:0*
T0*
_output_shapes	
:ð2'
%batch_normalization_1/batchnorm/mul_2Õ
#batch_normalization_1/batchnorm/subSub1batch_normalization_1/Cast/ReadVariableOp:value:0)batch_normalization_1/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:ð2%
#batch_normalization_1/batchnorm/subÞ
%batch_normalization_1/batchnorm/add_1AddV2)batch_normalization_1/batchnorm/mul_1:z:0'batch_normalization_1/batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿð2'
%batch_normalization_1/batchnorm/add_1v
Relu_2Relu)batch_normalization_1/batchnorm/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿð2
Relu_2¦
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	ðT*
dtype02
dense_1/MatMul/ReadVariableOp
dense_1/MatMulMatMulRelu_2:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT2
dense_1/MatMul¤
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:T*
dtype02 
dense_1/BiasAdd/ReadVariableOp¡
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT2
dense_1/BiasAddp
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT2
dense_1/Relu¶
4batch_normalization_2/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 26
4batch_normalization_2/moments/mean/reduction_indiceså
"batch_normalization_2/moments/meanMeandense_1/Relu:activations:0=batch_normalization_2/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:T*
	keep_dims(2$
"batch_normalization_2/moments/mean¾
*batch_normalization_2/moments/StopGradientStopGradient+batch_normalization_2/moments/mean:output:0*
T0*
_output_shapes

:T2,
*batch_normalization_2/moments/StopGradientú
/batch_normalization_2/moments/SquaredDifferenceSquaredDifferencedense_1/Relu:activations:03batch_normalization_2/moments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT21
/batch_normalization_2/moments/SquaredDifference¾
8batch_normalization_2/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2:
8batch_normalization_2/moments/variance/reduction_indices
&batch_normalization_2/moments/varianceMean3batch_normalization_2/moments/SquaredDifference:z:0Abatch_normalization_2/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:T*
	keep_dims(2(
&batch_normalization_2/moments/varianceÂ
%batch_normalization_2/moments/SqueezeSqueeze+batch_normalization_2/moments/mean:output:0*
T0*
_output_shapes
:T*
squeeze_dims
 2'
%batch_normalization_2/moments/SqueezeÊ
'batch_normalization_2/moments/Squeeze_1Squeeze/batch_normalization_2/moments/variance:output:0*
T0*
_output_shapes
:T*
squeeze_dims
 2)
'batch_normalization_2/moments/Squeeze_1Þ
+batch_normalization_2/AssignMovingAvg/decayConst*=
_class3
1/loc:@batch_normalization_2/AssignMovingAvg/5978*
_output_shapes
: *
dtype0*
valueB
 *
×#<2-
+batch_normalization_2/AssignMovingAvg/decayÓ
4batch_normalization_2/AssignMovingAvg/ReadVariableOpReadVariableOp*batch_normalization_2_assignmovingavg_5978*
_output_shapes
:T*
dtype026
4batch_normalization_2/AssignMovingAvg/ReadVariableOp¯
)batch_normalization_2/AssignMovingAvg/subSub<batch_normalization_2/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_2/moments/Squeeze:output:0*
T0*=
_class3
1/loc:@batch_normalization_2/AssignMovingAvg/5978*
_output_shapes
:T2+
)batch_normalization_2/AssignMovingAvg/sub¦
)batch_normalization_2/AssignMovingAvg/mulMul-batch_normalization_2/AssignMovingAvg/sub:z:04batch_normalization_2/AssignMovingAvg/decay:output:0*
T0*=
_class3
1/loc:@batch_normalization_2/AssignMovingAvg/5978*
_output_shapes
:T2+
)batch_normalization_2/AssignMovingAvg/mul
9batch_normalization_2/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp*batch_normalization_2_assignmovingavg_5978-batch_normalization_2/AssignMovingAvg/mul:z:05^batch_normalization_2/AssignMovingAvg/ReadVariableOp*=
_class3
1/loc:@batch_normalization_2/AssignMovingAvg/5978*
_output_shapes
 *
dtype02;
9batch_normalization_2/AssignMovingAvg/AssignSubVariableOpä
-batch_normalization_2/AssignMovingAvg_1/decayConst*?
_class5
31loc:@batch_normalization_2/AssignMovingAvg_1/5984*
_output_shapes
: *
dtype0*
valueB
 *
×#<2/
-batch_normalization_2/AssignMovingAvg_1/decayÙ
6batch_normalization_2/AssignMovingAvg_1/ReadVariableOpReadVariableOp,batch_normalization_2_assignmovingavg_1_5984*
_output_shapes
:T*
dtype028
6batch_normalization_2/AssignMovingAvg_1/ReadVariableOp¹
+batch_normalization_2/AssignMovingAvg_1/subSub>batch_normalization_2/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_2/moments/Squeeze_1:output:0*
T0*?
_class5
31loc:@batch_normalization_2/AssignMovingAvg_1/5984*
_output_shapes
:T2-
+batch_normalization_2/AssignMovingAvg_1/sub°
+batch_normalization_2/AssignMovingAvg_1/mulMul/batch_normalization_2/AssignMovingAvg_1/sub:z:06batch_normalization_2/AssignMovingAvg_1/decay:output:0*
T0*?
_class5
31loc:@batch_normalization_2/AssignMovingAvg_1/5984*
_output_shapes
:T2-
+batch_normalization_2/AssignMovingAvg_1/mul
;batch_normalization_2/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp,batch_normalization_2_assignmovingavg_1_5984/batch_normalization_2/AssignMovingAvg_1/mul:z:07^batch_normalization_2/AssignMovingAvg_1/ReadVariableOp*?
_class5
31loc:@batch_normalization_2/AssignMovingAvg_1/5984*
_output_shapes
 *
dtype02=
;batch_normalization_2/AssignMovingAvg_1/AssignSubVariableOpÅ
)batch_normalization_2/Cast/ReadVariableOpReadVariableOp2batch_normalization_2_cast_readvariableop_resource*
_output_shapes
:T*
dtype02+
)batch_normalization_2/Cast/ReadVariableOpË
+batch_normalization_2/Cast_1/ReadVariableOpReadVariableOp4batch_normalization_2_cast_1_readvariableop_resource*
_output_shapes
:T*
dtype02-
+batch_normalization_2/Cast_1/ReadVariableOp
%batch_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2'
%batch_normalization_2/batchnorm/add/yÚ
#batch_normalization_2/batchnorm/addAddV20batch_normalization_2/moments/Squeeze_1:output:0.batch_normalization_2/batchnorm/add/y:output:0*
T0*
_output_shapes
:T2%
#batch_normalization_2/batchnorm/add¥
%batch_normalization_2/batchnorm/RsqrtRsqrt'batch_normalization_2/batchnorm/add:z:0*
T0*
_output_shapes
:T2'
%batch_normalization_2/batchnorm/RsqrtÖ
#batch_normalization_2/batchnorm/mulMul)batch_normalization_2/batchnorm/Rsqrt:y:03batch_normalization_2/Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes
:T2%
#batch_normalization_2/batchnorm/mulÌ
%batch_normalization_2/batchnorm/mul_1Muldense_1/Relu:activations:0'batch_normalization_2/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT2'
%batch_normalization_2/batchnorm/mul_1Ó
%batch_normalization_2/batchnorm/mul_2Mul.batch_normalization_2/moments/Squeeze:output:0'batch_normalization_2/batchnorm/mul:z:0*
T0*
_output_shapes
:T2'
%batch_normalization_2/batchnorm/mul_2Ô
#batch_normalization_2/batchnorm/subSub1batch_normalization_2/Cast/ReadVariableOp:value:0)batch_normalization_2/batchnorm/mul_2:z:0*
T0*
_output_shapes
:T2%
#batch_normalization_2/batchnorm/subÝ
%batch_normalization_2/batchnorm/add_1AddV2)batch_normalization_2/batchnorm/mul_1:z:0'batch_normalization_2/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT2'
%batch_normalization_2/batchnorm/add_1z
SoftmaxSoftmax)batch_normalization_2/batchnorm/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT2	
Softmax¥
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

:T*
dtype02
dense_2/MatMul/ReadVariableOp
dense_2/MatMulMatMulSoftmax:softmax:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_2/MatMul¤
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_2/BiasAdd/ReadVariableOp¡
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_2/BiasAddy
dense_2/SoftmaxSoftmaxdense_2/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_2/Softmax­
IdentityIdentitydense_2/Softmax:softmax:0#^batch_normalization/AssignNewValue%^batch_normalization/AssignNewValue_1:^batch_normalization_1/AssignMovingAvg/AssignSubVariableOp<^batch_normalization_1/AssignMovingAvg_1/AssignSubVariableOp:^batch_normalization_2/AssignMovingAvg/AssignSubVariableOp<^batch_normalization_2/AssignMovingAvg_1/AssignSubVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*
_input_shapes}
{:ÿÿÿÿÿÿÿÿÿ  ::::::::::::::::::::::::2H
"batch_normalization/AssignNewValue"batch_normalization/AssignNewValue2L
$batch_normalization/AssignNewValue_1$batch_normalization/AssignNewValue_12v
9batch_normalization_1/AssignMovingAvg/AssignSubVariableOp9batch_normalization_1/AssignMovingAvg/AssignSubVariableOp2z
;batch_normalization_1/AssignMovingAvg_1/AssignSubVariableOp;batch_normalization_1/AssignMovingAvg_1/AssignSubVariableOp2v
9batch_normalization_2/AssignMovingAvg/AssignSubVariableOp9batch_normalization_2/AssignMovingAvg/AssignSubVariableOp2z
;batch_normalization_2/AssignMovingAvg_1/AssignSubVariableOp;batch_normalization_2/AssignMovingAvg_1/AssignSubVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
 
_user_specified_nameinputs
Î
Ø
,__inference_le_net_fcb_n3_layer_call_fn_6168

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22
identity¢StatefulPartitionedCall¢
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22*$
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*:
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_le_net_fcb_n3_layer_call_and_return_conditional_losses_56632
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*
_input_shapes}
{:ÿÿÿÿÿÿÿÿÿ  ::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
 
_user_specified_nameinputs
©
J
.__inference_max_pooling2d_1_layer_call_fn_4649

inputs
identityê
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_46432
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
	
ª
B__inference_conv2d_1_layer_call_and_return_conditional_losses_5194

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
Conv2D/ReadVariableOp¤
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ :::W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
íR
	
G__inference_le_net_fcb_n3_layer_call_and_return_conditional_losses_5516
input_1
conv2d_5103
conv2d_5105
batch_normalization_5172
batch_normalization_5174
batch_normalization_5176
batch_normalization_5178
conv2d_1_5205
conv2d_1_5207
conv2d_2_5271
conv2d_2_5273

dense_5354

dense_5356
batch_normalization_1_5415
batch_normalization_1_5417
batch_normalization_1_5419
batch_normalization_1_5421
dense_1_5447
dense_1_5449
batch_normalization_2_5478
batch_normalization_2_5480
batch_normalization_2_5482
batch_normalization_2_5484
dense_2_5510
dense_2_5512
identity¢+batch_normalization/StatefulPartitionedCall¢-batch_normalization_1/StatefulPartitionedCall¢-batch_normalization_2/StatefulPartitionedCall¢conv2d/StatefulPartitionedCall¢ conv2d_1/StatefulPartitionedCall¢ conv2d_2/StatefulPartitionedCall¢dense/StatefulPartitionedCall¢dense_1/StatefulPartitionedCall¢dense_2/StatefulPartitionedCall¢dropout/StatefulPartitionedCall¢)spatial_dropout2d/StatefulPartitionedCall¢+spatial_dropout2d_1/StatefulPartitionedCall
conv2d/StatefulPartitionedCallStatefulPartitionedCallinput_1conv2d_5103conv2d_5105*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_conv2d_layer_call_and_return_conditional_losses_50922 
conv2d/StatefulPartitionedCall¤
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0batch_normalization_5172batch_normalization_5174batch_normalization_5176batch_normalization_5178*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_batch_normalization_layer_call_and_return_conditional_losses_51272-
+batch_normalization/StatefulPartitionedCall
ReluRelu4batch_normalization/StatefulPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Reluõ
max_pooling2d/PartitionedCallPartitionedCallRelu:activations:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_46312
max_pooling2d/PartitionedCall¶
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0conv2d_1_5205conv2d_1_5207*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_conv2d_1_layer_call_and_return_conditional_losses_51942"
 conv2d_1/StatefulPartitionedCall°
)spatial_dropout2d/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_spatial_dropout2d_layer_call_and_return_conditional_losses_52322+
)spatial_dropout2d/StatefulPartitionedCallÃ
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall2spatial_dropout2d/StatefulPartitionedCall:output:0conv2d_2_5271conv2d_2_5273*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_conv2d_2_layer_call_and_return_conditional_losses_52602"
 conv2d_2/StatefulPartitionedCallã
+spatial_dropout2d_1/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*^spatial_dropout2d/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_spatial_dropout2d_1_layer_call_and_return_conditional_losses_52982-
+spatial_dropout2d_1/StatefulPartitionedCall
max_pooling2d_2/PartitionedCallPartitionedCall4spatial_dropout2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_47912!
max_pooling2d_2/PartitionedCall}
Relu_1Relu(max_pooling2d_2/PartitionedCall:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu_1þ
max_pooling2d_1/PartitionedCallPartitionedCallRelu_1:activations:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_46432!
max_pooling2d_1/PartitionedCallò
flatten/PartitionedCallPartitionedCall(max_pooling2d_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_flatten_layer_call_and_return_conditional_losses_53242
flatten/PartitionedCall
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0
dense_5354
dense_5356*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿð*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_53432
dense/StatefulPartitionedCall¶
dropout/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0,^spatial_dropout2d_1/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿð* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_53712!
dropout/StatefulPartitionedCall¬
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0batch_normalization_1_5415batch_normalization_1_5417batch_normalization_1_5419batch_normalization_1_5421*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿð*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_48932/
-batch_normalization_1/StatefulPartitionedCall
Relu_2Relu6batch_normalization_1/StatefulPartitionedCall:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿð2
Relu_2
dense_1/StatefulPartitionedCallStatefulPartitionedCallRelu_2:activations:0dense_1_5447dense_1_5449*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_54362!
dense_1/StatefulPartitionedCall«
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0batch_normalization_2_5478batch_normalization_2_5480batch_normalization_2_5482batch_normalization_2_5484*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_50332/
-batch_normalization_2/StatefulPartitionedCall
SoftmaxSoftmax6batch_normalization_2/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT2	
Softmax
dense_2/StatefulPartitionedCallStatefulPartitionedCallSoftmax:softmax:0dense_2_5510dense_2_5512*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dense_2_layer_call_and_return_conditional_losses_54992!
dense_2/StatefulPartitionedCallÑ
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dropout/StatefulPartitionedCall*^spatial_dropout2d/StatefulPartitionedCall,^spatial_dropout2d_1/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*
_input_shapes}
{:ÿÿÿÿÿÿÿÿÿ  ::::::::::::::::::::::::2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2V
)spatial_dropout2d/StatefulPartitionedCall)spatial_dropout2d/StatefulPartitionedCall2Z
+spatial_dropout2d_1/StatefulPartitionedCall+spatial_dropout2d_1/StatefulPartitionedCall:X T
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
!
_user_specified_name	input_1
 
Ù
__inference__wrapped_model_4521
input_17
3le_net_fcb_n3_conv2d_conv2d_readvariableop_resource8
4le_net_fcb_n3_conv2d_biasadd_readvariableop_resource=
9le_net_fcb_n3_batch_normalization_readvariableop_resource?
;le_net_fcb_n3_batch_normalization_readvariableop_1_resourceN
Jle_net_fcb_n3_batch_normalization_fusedbatchnormv3_readvariableop_resourceP
Lle_net_fcb_n3_batch_normalization_fusedbatchnormv3_readvariableop_1_resource9
5le_net_fcb_n3_conv2d_1_conv2d_readvariableop_resource:
6le_net_fcb_n3_conv2d_1_biasadd_readvariableop_resource9
5le_net_fcb_n3_conv2d_2_conv2d_readvariableop_resource:
6le_net_fcb_n3_conv2d_2_biasadd_readvariableop_resource6
2le_net_fcb_n3_dense_matmul_readvariableop_resource7
3le_net_fcb_n3_dense_biasadd_readvariableop_resourceD
@le_net_fcb_n3_batch_normalization_1_cast_readvariableop_resourceF
Ble_net_fcb_n3_batch_normalization_1_cast_1_readvariableop_resourceF
Ble_net_fcb_n3_batch_normalization_1_cast_2_readvariableop_resourceF
Ble_net_fcb_n3_batch_normalization_1_cast_3_readvariableop_resource8
4le_net_fcb_n3_dense_1_matmul_readvariableop_resource9
5le_net_fcb_n3_dense_1_biasadd_readvariableop_resourceD
@le_net_fcb_n3_batch_normalization_2_cast_readvariableop_resourceF
Ble_net_fcb_n3_batch_normalization_2_cast_1_readvariableop_resourceF
Ble_net_fcb_n3_batch_normalization_2_cast_2_readvariableop_resourceF
Ble_net_fcb_n3_batch_normalization_2_cast_3_readvariableop_resource8
4le_net_fcb_n3_dense_2_matmul_readvariableop_resource9
5le_net_fcb_n3_dense_2_biasadd_readvariableop_resource
identityÔ
*le_net_fcb_n3/conv2d/Conv2D/ReadVariableOpReadVariableOp3le_net_fcb_n3_conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02,
*le_net_fcb_n3/conv2d/Conv2D/ReadVariableOpä
le_net_fcb_n3/conv2d/Conv2DConv2Dinput_12le_net_fcb_n3/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingVALID*
strides
2
le_net_fcb_n3/conv2d/Conv2DË
+le_net_fcb_n3/conv2d/BiasAdd/ReadVariableOpReadVariableOp4le_net_fcb_n3_conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02-
+le_net_fcb_n3/conv2d/BiasAdd/ReadVariableOpÜ
le_net_fcb_n3/conv2d/BiasAddBiasAdd$le_net_fcb_n3/conv2d/Conv2D:output:03le_net_fcb_n3/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
le_net_fcb_n3/conv2d/BiasAdd
le_net_fcb_n3/conv2d/ReluRelu%le_net_fcb_n3/conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
le_net_fcb_n3/conv2d/ReluÚ
0le_net_fcb_n3/batch_normalization/ReadVariableOpReadVariableOp9le_net_fcb_n3_batch_normalization_readvariableop_resource*
_output_shapes
: *
dtype022
0le_net_fcb_n3/batch_normalization/ReadVariableOpà
2le_net_fcb_n3/batch_normalization/ReadVariableOp_1ReadVariableOp;le_net_fcb_n3_batch_normalization_readvariableop_1_resource*
_output_shapes
: *
dtype024
2le_net_fcb_n3/batch_normalization/ReadVariableOp_1
Ale_net_fcb_n3/batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOpJle_net_fcb_n3_batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02C
Ale_net_fcb_n3/batch_normalization/FusedBatchNormV3/ReadVariableOp
Cle_net_fcb_n3/batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpLle_net_fcb_n3_batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02E
Cle_net_fcb_n3/batch_normalization/FusedBatchNormV3/ReadVariableOp_1·
2le_net_fcb_n3/batch_normalization/FusedBatchNormV3FusedBatchNormV3'le_net_fcb_n3/conv2d/Relu:activations:08le_net_fcb_n3/batch_normalization/ReadVariableOp:value:0:le_net_fcb_n3/batch_normalization/ReadVariableOp_1:value:0Ile_net_fcb_n3/batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0Kle_net_fcb_n3/batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
is_training( 24
2le_net_fcb_n3/batch_normalization/FusedBatchNormV3¢
le_net_fcb_n3/ReluRelu6le_net_fcb_n3/batch_normalization/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
le_net_fcb_n3/Reluä
#le_net_fcb_n3/max_pooling2d/MaxPoolMaxPool le_net_fcb_n3/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
ksize
*
paddingVALID*
strides
2%
#le_net_fcb_n3/max_pooling2d/MaxPoolÚ
,le_net_fcb_n3/conv2d_1/Conv2D/ReadVariableOpReadVariableOp5le_net_fcb_n3_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02.
,le_net_fcb_n3/conv2d_1/Conv2D/ReadVariableOp
le_net_fcb_n3/conv2d_1/Conv2DConv2D,le_net_fcb_n3/max_pooling2d/MaxPool:output:04le_net_fcb_n3/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
2
le_net_fcb_n3/conv2d_1/Conv2DÑ
-le_net_fcb_n3/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp6le_net_fcb_n3_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-le_net_fcb_n3/conv2d_1/BiasAdd/ReadVariableOpä
le_net_fcb_n3/conv2d_1/BiasAddBiasAdd&le_net_fcb_n3/conv2d_1/Conv2D:output:05le_net_fcb_n3/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2 
le_net_fcb_n3/conv2d_1/BiasAdd¥
le_net_fcb_n3/conv2d_1/ReluRelu'le_net_fcb_n3/conv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
le_net_fcb_n3/conv2d_1/ReluÅ
(le_net_fcb_n3/spatial_dropout2d/IdentityIdentity)le_net_fcb_n3/conv2d_1/Relu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2*
(le_net_fcb_n3/spatial_dropout2d/IdentityÛ
,le_net_fcb_n3/conv2d_2/Conv2D/ReadVariableOpReadVariableOp5le_net_fcb_n3_conv2d_2_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02.
,le_net_fcb_n3/conv2d_2/Conv2D/ReadVariableOp
le_net_fcb_n3/conv2d_2/Conv2DConv2D1le_net_fcb_n3/spatial_dropout2d/Identity:output:04le_net_fcb_n3/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2
le_net_fcb_n3/conv2d_2/Conv2DÒ
-le_net_fcb_n3/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp6le_net_fcb_n3_conv2d_2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02/
-le_net_fcb_n3/conv2d_2/BiasAdd/ReadVariableOpå
le_net_fcb_n3/conv2d_2/BiasAddBiasAdd&le_net_fcb_n3/conv2d_2/Conv2D:output:05le_net_fcb_n3/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
le_net_fcb_n3/conv2d_2/BiasAdd¦
le_net_fcb_n3/conv2d_2/ReluRelu'le_net_fcb_n3/conv2d_2/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
le_net_fcb_n3/conv2d_2/ReluÊ
*le_net_fcb_n3/spatial_dropout2d_1/IdentityIdentity)le_net_fcb_n3/conv2d_2/Relu:activations:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2,
*le_net_fcb_n3/spatial_dropout2d_1/Identityü
%le_net_fcb_n3/max_pooling2d_2/MaxPoolMaxPool3le_net_fcb_n3/spatial_dropout2d_1/Identity:output:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
2'
%le_net_fcb_n3/max_pooling2d_2/MaxPool
le_net_fcb_n3/Relu_1Relu.le_net_fcb_n3/max_pooling2d_2/MaxPool:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
le_net_fcb_n3/Relu_1ë
%le_net_fcb_n3/max_pooling2d_1/MaxPoolMaxPool"le_net_fcb_n3/Relu_1:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
2'
%le_net_fcb_n3/max_pooling2d_1/MaxPool
le_net_fcb_n3/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
le_net_fcb_n3/flatten/ConstÒ
le_net_fcb_n3/flatten/ReshapeReshape.le_net_fcb_n3/max_pooling2d_1/MaxPool:output:0$le_net_fcb_n3/flatten/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
le_net_fcb_n3/flatten/ReshapeË
)le_net_fcb_n3/dense/MatMul/ReadVariableOpReadVariableOp2le_net_fcb_n3_dense_matmul_readvariableop_resource* 
_output_shapes
:
ð*
dtype02+
)le_net_fcb_n3/dense/MatMul/ReadVariableOpÐ
le_net_fcb_n3/dense/MatMulMatMul&le_net_fcb_n3/flatten/Reshape:output:01le_net_fcb_n3/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿð2
le_net_fcb_n3/dense/MatMulÉ
*le_net_fcb_n3/dense/BiasAdd/ReadVariableOpReadVariableOp3le_net_fcb_n3_dense_biasadd_readvariableop_resource*
_output_shapes	
:ð*
dtype02,
*le_net_fcb_n3/dense/BiasAdd/ReadVariableOpÒ
le_net_fcb_n3/dense/BiasAddBiasAdd$le_net_fcb_n3/dense/MatMul:product:02le_net_fcb_n3/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿð2
le_net_fcb_n3/dense/BiasAdd
le_net_fcb_n3/dense/ReluRelu$le_net_fcb_n3/dense/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿð2
le_net_fcb_n3/dense/Relu§
le_net_fcb_n3/dropout/IdentityIdentity&le_net_fcb_n3/dense/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿð2 
le_net_fcb_n3/dropout/Identityð
7le_net_fcb_n3/batch_normalization_1/Cast/ReadVariableOpReadVariableOp@le_net_fcb_n3_batch_normalization_1_cast_readvariableop_resource*
_output_shapes	
:ð*
dtype029
7le_net_fcb_n3/batch_normalization_1/Cast/ReadVariableOpö
9le_net_fcb_n3/batch_normalization_1/Cast_1/ReadVariableOpReadVariableOpBle_net_fcb_n3_batch_normalization_1_cast_1_readvariableop_resource*
_output_shapes	
:ð*
dtype02;
9le_net_fcb_n3/batch_normalization_1/Cast_1/ReadVariableOpö
9le_net_fcb_n3/batch_normalization_1/Cast_2/ReadVariableOpReadVariableOpBle_net_fcb_n3_batch_normalization_1_cast_2_readvariableop_resource*
_output_shapes	
:ð*
dtype02;
9le_net_fcb_n3/batch_normalization_1/Cast_2/ReadVariableOpö
9le_net_fcb_n3/batch_normalization_1/Cast_3/ReadVariableOpReadVariableOpBle_net_fcb_n3_batch_normalization_1_cast_3_readvariableop_resource*
_output_shapes	
:ð*
dtype02;
9le_net_fcb_n3/batch_normalization_1/Cast_3/ReadVariableOp¯
3le_net_fcb_n3/batch_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:25
3le_net_fcb_n3/batch_normalization_1/batchnorm/add/y
1le_net_fcb_n3/batch_normalization_1/batchnorm/addAddV2Ale_net_fcb_n3/batch_normalization_1/Cast_1/ReadVariableOp:value:0<le_net_fcb_n3/batch_normalization_1/batchnorm/add/y:output:0*
T0*
_output_shapes	
:ð23
1le_net_fcb_n3/batch_normalization_1/batchnorm/addÐ
3le_net_fcb_n3/batch_normalization_1/batchnorm/RsqrtRsqrt5le_net_fcb_n3/batch_normalization_1/batchnorm/add:z:0*
T0*
_output_shapes	
:ð25
3le_net_fcb_n3/batch_normalization_1/batchnorm/Rsqrt
1le_net_fcb_n3/batch_normalization_1/batchnorm/mulMul7le_net_fcb_n3/batch_normalization_1/batchnorm/Rsqrt:y:0Ale_net_fcb_n3/batch_normalization_1/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:ð23
1le_net_fcb_n3/batch_normalization_1/batchnorm/mul
3le_net_fcb_n3/batch_normalization_1/batchnorm/mul_1Mul'le_net_fcb_n3/dropout/Identity:output:05le_net_fcb_n3/batch_normalization_1/batchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿð25
3le_net_fcb_n3/batch_normalization_1/batchnorm/mul_1
3le_net_fcb_n3/batch_normalization_1/batchnorm/mul_2Mul?le_net_fcb_n3/batch_normalization_1/Cast/ReadVariableOp:value:05le_net_fcb_n3/batch_normalization_1/batchnorm/mul:z:0*
T0*
_output_shapes	
:ð25
3le_net_fcb_n3/batch_normalization_1/batchnorm/mul_2
1le_net_fcb_n3/batch_normalization_1/batchnorm/subSubAle_net_fcb_n3/batch_normalization_1/Cast_2/ReadVariableOp:value:07le_net_fcb_n3/batch_normalization_1/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:ð23
1le_net_fcb_n3/batch_normalization_1/batchnorm/sub
3le_net_fcb_n3/batch_normalization_1/batchnorm/add_1AddV27le_net_fcb_n3/batch_normalization_1/batchnorm/mul_1:z:05le_net_fcb_n3/batch_normalization_1/batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿð25
3le_net_fcb_n3/batch_normalization_1/batchnorm/add_1 
le_net_fcb_n3/Relu_2Relu7le_net_fcb_n3/batch_normalization_1/batchnorm/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿð2
le_net_fcb_n3/Relu_2Ð
+le_net_fcb_n3/dense_1/MatMul/ReadVariableOpReadVariableOp4le_net_fcb_n3_dense_1_matmul_readvariableop_resource*
_output_shapes
:	ðT*
dtype02-
+le_net_fcb_n3/dense_1/MatMul/ReadVariableOpÑ
le_net_fcb_n3/dense_1/MatMulMatMul"le_net_fcb_n3/Relu_2:activations:03le_net_fcb_n3/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT2
le_net_fcb_n3/dense_1/MatMulÎ
,le_net_fcb_n3/dense_1/BiasAdd/ReadVariableOpReadVariableOp5le_net_fcb_n3_dense_1_biasadd_readvariableop_resource*
_output_shapes
:T*
dtype02.
,le_net_fcb_n3/dense_1/BiasAdd/ReadVariableOpÙ
le_net_fcb_n3/dense_1/BiasAddBiasAdd&le_net_fcb_n3/dense_1/MatMul:product:04le_net_fcb_n3/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT2
le_net_fcb_n3/dense_1/BiasAdd
le_net_fcb_n3/dense_1/ReluRelu&le_net_fcb_n3/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT2
le_net_fcb_n3/dense_1/Reluï
7le_net_fcb_n3/batch_normalization_2/Cast/ReadVariableOpReadVariableOp@le_net_fcb_n3_batch_normalization_2_cast_readvariableop_resource*
_output_shapes
:T*
dtype029
7le_net_fcb_n3/batch_normalization_2/Cast/ReadVariableOpõ
9le_net_fcb_n3/batch_normalization_2/Cast_1/ReadVariableOpReadVariableOpBle_net_fcb_n3_batch_normalization_2_cast_1_readvariableop_resource*
_output_shapes
:T*
dtype02;
9le_net_fcb_n3/batch_normalization_2/Cast_1/ReadVariableOpõ
9le_net_fcb_n3/batch_normalization_2/Cast_2/ReadVariableOpReadVariableOpBle_net_fcb_n3_batch_normalization_2_cast_2_readvariableop_resource*
_output_shapes
:T*
dtype02;
9le_net_fcb_n3/batch_normalization_2/Cast_2/ReadVariableOpõ
9le_net_fcb_n3/batch_normalization_2/Cast_3/ReadVariableOpReadVariableOpBle_net_fcb_n3_batch_normalization_2_cast_3_readvariableop_resource*
_output_shapes
:T*
dtype02;
9le_net_fcb_n3/batch_normalization_2/Cast_3/ReadVariableOp¯
3le_net_fcb_n3/batch_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:25
3le_net_fcb_n3/batch_normalization_2/batchnorm/add/y
1le_net_fcb_n3/batch_normalization_2/batchnorm/addAddV2Ale_net_fcb_n3/batch_normalization_2/Cast_1/ReadVariableOp:value:0<le_net_fcb_n3/batch_normalization_2/batchnorm/add/y:output:0*
T0*
_output_shapes
:T23
1le_net_fcb_n3/batch_normalization_2/batchnorm/addÏ
3le_net_fcb_n3/batch_normalization_2/batchnorm/RsqrtRsqrt5le_net_fcb_n3/batch_normalization_2/batchnorm/add:z:0*
T0*
_output_shapes
:T25
3le_net_fcb_n3/batch_normalization_2/batchnorm/Rsqrt
1le_net_fcb_n3/batch_normalization_2/batchnorm/mulMul7le_net_fcb_n3/batch_normalization_2/batchnorm/Rsqrt:y:0Ale_net_fcb_n3/batch_normalization_2/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:T23
1le_net_fcb_n3/batch_normalization_2/batchnorm/mul
3le_net_fcb_n3/batch_normalization_2/batchnorm/mul_1Mul(le_net_fcb_n3/dense_1/Relu:activations:05le_net_fcb_n3/batch_normalization_2/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT25
3le_net_fcb_n3/batch_normalization_2/batchnorm/mul_1
3le_net_fcb_n3/batch_normalization_2/batchnorm/mul_2Mul?le_net_fcb_n3/batch_normalization_2/Cast/ReadVariableOp:value:05le_net_fcb_n3/batch_normalization_2/batchnorm/mul:z:0*
T0*
_output_shapes
:T25
3le_net_fcb_n3/batch_normalization_2/batchnorm/mul_2
1le_net_fcb_n3/batch_normalization_2/batchnorm/subSubAle_net_fcb_n3/batch_normalization_2/Cast_2/ReadVariableOp:value:07le_net_fcb_n3/batch_normalization_2/batchnorm/mul_2:z:0*
T0*
_output_shapes
:T23
1le_net_fcb_n3/batch_normalization_2/batchnorm/sub
3le_net_fcb_n3/batch_normalization_2/batchnorm/add_1AddV27le_net_fcb_n3/batch_normalization_2/batchnorm/mul_1:z:05le_net_fcb_n3/batch_normalization_2/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT25
3le_net_fcb_n3/batch_normalization_2/batchnorm/add_1¤
le_net_fcb_n3/SoftmaxSoftmax7le_net_fcb_n3/batch_normalization_2/batchnorm/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT2
le_net_fcb_n3/SoftmaxÏ
+le_net_fcb_n3/dense_2/MatMul/ReadVariableOpReadVariableOp4le_net_fcb_n3_dense_2_matmul_readvariableop_resource*
_output_shapes

:T*
dtype02-
+le_net_fcb_n3/dense_2/MatMul/ReadVariableOpÎ
le_net_fcb_n3/dense_2/MatMulMatMulle_net_fcb_n3/Softmax:softmax:03le_net_fcb_n3/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
le_net_fcb_n3/dense_2/MatMulÎ
,le_net_fcb_n3/dense_2/BiasAdd/ReadVariableOpReadVariableOp5le_net_fcb_n3_dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02.
,le_net_fcb_n3/dense_2/BiasAdd/ReadVariableOpÙ
le_net_fcb_n3/dense_2/BiasAddBiasAdd&le_net_fcb_n3/dense_2/MatMul:product:04le_net_fcb_n3/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
le_net_fcb_n3/dense_2/BiasAdd£
le_net_fcb_n3/dense_2/SoftmaxSoftmax&le_net_fcb_n3/dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
le_net_fcb_n3/dense_2/Softmax{
IdentityIdentity'le_net_fcb_n3/dense_2/Softmax:softmax:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*
_input_shapes}
{:ÿÿÿÿÿÿÿÿÿ  :::::::::::::::::::::::::X T
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
!
_user_specified_name	input_1
ô
z
%__inference_conv2d_layer_call_fn_6241

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallø
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_conv2d_layer_call_and_return_conditional_losses_50922
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ  ::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
 
_user_specified_nameinputs
»
]
A__inference_flatten_layer_call_and_return_conditional_losses_5324

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ã(
¸
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_5033

inputs
assignmovingavg_5008
assignmovingavg_1_5014 
cast_readvariableop_resource"
cast_1_readvariableop_resource
identity¢#AssignMovingAvg/AssignSubVariableOp¢%AssignMovingAvg_1/AssignSubVariableOp
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indices
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:T*
	keep_dims(2
moments/mean|
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:T2
moments/StopGradient¤
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT2
moments/SquaredDifference
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices²
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:T*
	keep_dims(2
moments/variance
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:T*
squeeze_dims
 2
moments/Squeeze
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:T*
squeeze_dims
 2
moments/Squeeze_1
AssignMovingAvg/decayConst*'
_class
loc:@AssignMovingAvg/5008*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg/decay
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_5008*
_output_shapes
:T*
dtype02 
AssignMovingAvg/ReadVariableOpÁ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*'
_class
loc:@AssignMovingAvg/5008*
_output_shapes
:T2
AssignMovingAvg/sub¸
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*'
_class
loc:@AssignMovingAvg/5008*
_output_shapes
:T2
AssignMovingAvg/mulý
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_5008AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*'
_class
loc:@AssignMovingAvg/5008*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp¢
AssignMovingAvg_1/decayConst*)
_class
loc:@AssignMovingAvg_1/5014*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg_1/decay
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_5014*
_output_shapes
:T*
dtype02"
 AssignMovingAvg_1/ReadVariableOpË
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*)
_class
loc:@AssignMovingAvg_1/5014*
_output_shapes
:T2
AssignMovingAvg_1/subÂ
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*)
_class
loc:@AssignMovingAvg_1/5014*
_output_shapes
:T2
AssignMovingAvg_1/mul
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_5014AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*)
_class
loc:@AssignMovingAvg_1/5014*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:T*
dtype02
Cast/ReadVariableOp
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
:T*
dtype02
Cast_1/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:T2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:T2
batchnorm/Rsqrt~
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes
:T2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:T2
batchnorm/mul_2|
batchnorm/subSubCast/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:T2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT2
batchnorm/add_1µ
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿT::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT
 
_user_specified_nameinputs
­
L
0__inference_spatial_dropout2d_layer_call_fn_6465

inputs
identityì
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_spatial_dropout2d_layer_call_and_return_conditional_losses_47142
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
£
j
K__inference_spatial_dropout2d_layer_call_and_return_conditional_losses_4704

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ì
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/Const
dropout/MulMulinputsdropout/Const:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dropout/Mul
dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2 
dropout/random_uniform/shape/1
dropout/random_uniform/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2 
dropout/random_uniform/shape/2ö
dropout/random_uniform/shapePackstrided_slice:output:0'dropout/random_uniform/shape/1:output:0'dropout/random_uniform/shape/2:output:0strided_slice_1:output:0*
N*
T0*
_output_shapes
:2
dropout/random_uniform/shapeÔ
$dropout/random_uniform/RandomUniformRandomUniform%dropout/random_uniform/shape:output:0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2
dropout/GreaterEqual/yÏ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dropout/Mul_1
IdentityIdentitydropout/Mul_1:z:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¦
Ó$
__inference__traced_save_7053
file_prefix:
6savev2_le_net_fcb_n3_conv2d_kernel_read_readvariableop8
4savev2_le_net_fcb_n3_conv2d_bias_read_readvariableopF
Bsavev2_le_net_fcb_n3_batch_normalization_gamma_read_readvariableopE
Asavev2_le_net_fcb_n3_batch_normalization_beta_read_readvariableopL
Hsavev2_le_net_fcb_n3_batch_normalization_moving_mean_read_readvariableopP
Lsavev2_le_net_fcb_n3_batch_normalization_moving_variance_read_readvariableop<
8savev2_le_net_fcb_n3_conv2d_1_kernel_read_readvariableop:
6savev2_le_net_fcb_n3_conv2d_1_bias_read_readvariableop<
8savev2_le_net_fcb_n3_conv2d_2_kernel_read_readvariableop:
6savev2_le_net_fcb_n3_conv2d_2_bias_read_readvariableop9
5savev2_le_net_fcb_n3_dense_kernel_read_readvariableop7
3savev2_le_net_fcb_n3_dense_bias_read_readvariableopH
Dsavev2_le_net_fcb_n3_batch_normalization_1_gamma_read_readvariableopG
Csavev2_le_net_fcb_n3_batch_normalization_1_beta_read_readvariableopN
Jsavev2_le_net_fcb_n3_batch_normalization_1_moving_mean_read_readvariableopR
Nsavev2_le_net_fcb_n3_batch_normalization_1_moving_variance_read_readvariableopH
Dsavev2_le_net_fcb_n3_batch_normalization_2_gamma_read_readvariableopG
Csavev2_le_net_fcb_n3_batch_normalization_2_beta_read_readvariableopN
Jsavev2_le_net_fcb_n3_batch_normalization_2_moving_mean_read_readvariableopR
Nsavev2_le_net_fcb_n3_batch_normalization_2_moving_variance_read_readvariableop;
7savev2_le_net_fcb_n3_dense_1_kernel_read_readvariableop9
5savev2_le_net_fcb_n3_dense_1_bias_read_readvariableop;
7savev2_le_net_fcb_n3_dense_2_kernel_read_readvariableop9
5savev2_le_net_fcb_n3_dense_2_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableopA
=savev2_adam_le_net_fcb_n3_conv2d_kernel_m_read_readvariableop?
;savev2_adam_le_net_fcb_n3_conv2d_bias_m_read_readvariableopM
Isavev2_adam_le_net_fcb_n3_batch_normalization_gamma_m_read_readvariableopL
Hsavev2_adam_le_net_fcb_n3_batch_normalization_beta_m_read_readvariableopC
?savev2_adam_le_net_fcb_n3_conv2d_1_kernel_m_read_readvariableopA
=savev2_adam_le_net_fcb_n3_conv2d_1_bias_m_read_readvariableopC
?savev2_adam_le_net_fcb_n3_conv2d_2_kernel_m_read_readvariableopA
=savev2_adam_le_net_fcb_n3_conv2d_2_bias_m_read_readvariableop@
<savev2_adam_le_net_fcb_n3_dense_kernel_m_read_readvariableop>
:savev2_adam_le_net_fcb_n3_dense_bias_m_read_readvariableopO
Ksavev2_adam_le_net_fcb_n3_batch_normalization_1_gamma_m_read_readvariableopN
Jsavev2_adam_le_net_fcb_n3_batch_normalization_1_beta_m_read_readvariableopO
Ksavev2_adam_le_net_fcb_n3_batch_normalization_2_gamma_m_read_readvariableopN
Jsavev2_adam_le_net_fcb_n3_batch_normalization_2_beta_m_read_readvariableopB
>savev2_adam_le_net_fcb_n3_dense_1_kernel_m_read_readvariableop@
<savev2_adam_le_net_fcb_n3_dense_1_bias_m_read_readvariableopB
>savev2_adam_le_net_fcb_n3_dense_2_kernel_m_read_readvariableop@
<savev2_adam_le_net_fcb_n3_dense_2_bias_m_read_readvariableopA
=savev2_adam_le_net_fcb_n3_conv2d_kernel_v_read_readvariableop?
;savev2_adam_le_net_fcb_n3_conv2d_bias_v_read_readvariableopM
Isavev2_adam_le_net_fcb_n3_batch_normalization_gamma_v_read_readvariableopL
Hsavev2_adam_le_net_fcb_n3_batch_normalization_beta_v_read_readvariableopC
?savev2_adam_le_net_fcb_n3_conv2d_1_kernel_v_read_readvariableopA
=savev2_adam_le_net_fcb_n3_conv2d_1_bias_v_read_readvariableopC
?savev2_adam_le_net_fcb_n3_conv2d_2_kernel_v_read_readvariableopA
=savev2_adam_le_net_fcb_n3_conv2d_2_bias_v_read_readvariableop@
<savev2_adam_le_net_fcb_n3_dense_kernel_v_read_readvariableop>
:savev2_adam_le_net_fcb_n3_dense_bias_v_read_readvariableopO
Ksavev2_adam_le_net_fcb_n3_batch_normalization_1_gamma_v_read_readvariableopN
Jsavev2_adam_le_net_fcb_n3_batch_normalization_1_beta_v_read_readvariableopO
Ksavev2_adam_le_net_fcb_n3_batch_normalization_2_gamma_v_read_readvariableopN
Jsavev2_adam_le_net_fcb_n3_batch_normalization_2_beta_v_read_readvariableopB
>savev2_adam_le_net_fcb_n3_dense_1_kernel_v_read_readvariableop@
<savev2_adam_le_net_fcb_n3_dense_1_bias_v_read_readvariableopB
>savev2_adam_le_net_fcb_n3_dense_2_kernel_v_read_readvariableop@
<savev2_adam_le_net_fcb_n3_dense_2_bias_v_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpoints
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Const
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_4cb9225be00d4430a80a96d2c8827a1e/part2	
Const_1
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard¦
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameç
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:F*
dtype0*ù
valueïBìFB$c1/kernel/.ATTRIBUTES/VARIABLE_VALUEB"c1/bias/.ATTRIBUTES/VARIABLE_VALUEB(affine1/gamma/.ATTRIBUTES/VARIABLE_VALUEB'affine1/beta/.ATTRIBUTES/VARIABLE_VALUEB.affine1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB2affine1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB$c3/kernel/.ATTRIBUTES/VARIABLE_VALUEB"c3/bias/.ATTRIBUTES/VARIABLE_VALUEB'c_add/kernel/.ATTRIBUTES/VARIABLE_VALUEB%c_add/bias/.ATTRIBUTES/VARIABLE_VALUEB$c5/kernel/.ATTRIBUTES/VARIABLE_VALUEB"c5/bias/.ATTRIBUTES/VARIABLE_VALUEB(affine3/gamma/.ATTRIBUTES/VARIABLE_VALUEB'affine3/beta/.ATTRIBUTES/VARIABLE_VALUEB.affine3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB2affine3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB(affine4/gamma/.ATTRIBUTES/VARIABLE_VALUEB'affine4/beta/.ATTRIBUTES/VARIABLE_VALUEB.affine4/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB2affine4/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB$f6/kernel/.ATTRIBUTES/VARIABLE_VALUEB"f6/bias/.ATTRIBUTES/VARIABLE_VALUEB.output_layer/kernel/.ATTRIBUTES/VARIABLE_VALUEB,output_layer/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB@c1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB>c1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBDaffine1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCaffine1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB@c3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB>c3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCc_add/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBAc_add/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB@c5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB>c5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBDaffine3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCaffine3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBDaffine4/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCaffine4/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB@f6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB>f6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBJoutput_layer/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBHoutput_layer/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB@c1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB>c1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBDaffine1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCaffine1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB@c3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB>c3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCc_add/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBAc_add/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB@c5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB>c5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBDaffine3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCaffine3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBDaffine4/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCaffine4/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB@f6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB>f6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBJoutput_layer/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBHoutput_layer/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:F*
dtype0*¡
valueBFB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesÔ#
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:06savev2_le_net_fcb_n3_conv2d_kernel_read_readvariableop4savev2_le_net_fcb_n3_conv2d_bias_read_readvariableopBsavev2_le_net_fcb_n3_batch_normalization_gamma_read_readvariableopAsavev2_le_net_fcb_n3_batch_normalization_beta_read_readvariableopHsavev2_le_net_fcb_n3_batch_normalization_moving_mean_read_readvariableopLsavev2_le_net_fcb_n3_batch_normalization_moving_variance_read_readvariableop8savev2_le_net_fcb_n3_conv2d_1_kernel_read_readvariableop6savev2_le_net_fcb_n3_conv2d_1_bias_read_readvariableop8savev2_le_net_fcb_n3_conv2d_2_kernel_read_readvariableop6savev2_le_net_fcb_n3_conv2d_2_bias_read_readvariableop5savev2_le_net_fcb_n3_dense_kernel_read_readvariableop3savev2_le_net_fcb_n3_dense_bias_read_readvariableopDsavev2_le_net_fcb_n3_batch_normalization_1_gamma_read_readvariableopCsavev2_le_net_fcb_n3_batch_normalization_1_beta_read_readvariableopJsavev2_le_net_fcb_n3_batch_normalization_1_moving_mean_read_readvariableopNsavev2_le_net_fcb_n3_batch_normalization_1_moving_variance_read_readvariableopDsavev2_le_net_fcb_n3_batch_normalization_2_gamma_read_readvariableopCsavev2_le_net_fcb_n3_batch_normalization_2_beta_read_readvariableopJsavev2_le_net_fcb_n3_batch_normalization_2_moving_mean_read_readvariableopNsavev2_le_net_fcb_n3_batch_normalization_2_moving_variance_read_readvariableop7savev2_le_net_fcb_n3_dense_1_kernel_read_readvariableop5savev2_le_net_fcb_n3_dense_1_bias_read_readvariableop7savev2_le_net_fcb_n3_dense_2_kernel_read_readvariableop5savev2_le_net_fcb_n3_dense_2_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop=savev2_adam_le_net_fcb_n3_conv2d_kernel_m_read_readvariableop;savev2_adam_le_net_fcb_n3_conv2d_bias_m_read_readvariableopIsavev2_adam_le_net_fcb_n3_batch_normalization_gamma_m_read_readvariableopHsavev2_adam_le_net_fcb_n3_batch_normalization_beta_m_read_readvariableop?savev2_adam_le_net_fcb_n3_conv2d_1_kernel_m_read_readvariableop=savev2_adam_le_net_fcb_n3_conv2d_1_bias_m_read_readvariableop?savev2_adam_le_net_fcb_n3_conv2d_2_kernel_m_read_readvariableop=savev2_adam_le_net_fcb_n3_conv2d_2_bias_m_read_readvariableop<savev2_adam_le_net_fcb_n3_dense_kernel_m_read_readvariableop:savev2_adam_le_net_fcb_n3_dense_bias_m_read_readvariableopKsavev2_adam_le_net_fcb_n3_batch_normalization_1_gamma_m_read_readvariableopJsavev2_adam_le_net_fcb_n3_batch_normalization_1_beta_m_read_readvariableopKsavev2_adam_le_net_fcb_n3_batch_normalization_2_gamma_m_read_readvariableopJsavev2_adam_le_net_fcb_n3_batch_normalization_2_beta_m_read_readvariableop>savev2_adam_le_net_fcb_n3_dense_1_kernel_m_read_readvariableop<savev2_adam_le_net_fcb_n3_dense_1_bias_m_read_readvariableop>savev2_adam_le_net_fcb_n3_dense_2_kernel_m_read_readvariableop<savev2_adam_le_net_fcb_n3_dense_2_bias_m_read_readvariableop=savev2_adam_le_net_fcb_n3_conv2d_kernel_v_read_readvariableop;savev2_adam_le_net_fcb_n3_conv2d_bias_v_read_readvariableopIsavev2_adam_le_net_fcb_n3_batch_normalization_gamma_v_read_readvariableopHsavev2_adam_le_net_fcb_n3_batch_normalization_beta_v_read_readvariableop?savev2_adam_le_net_fcb_n3_conv2d_1_kernel_v_read_readvariableop=savev2_adam_le_net_fcb_n3_conv2d_1_bias_v_read_readvariableop?savev2_adam_le_net_fcb_n3_conv2d_2_kernel_v_read_readvariableop=savev2_adam_le_net_fcb_n3_conv2d_2_bias_v_read_readvariableop<savev2_adam_le_net_fcb_n3_dense_kernel_v_read_readvariableop:savev2_adam_le_net_fcb_n3_dense_bias_v_read_readvariableopKsavev2_adam_le_net_fcb_n3_batch_normalization_1_gamma_v_read_readvariableopJsavev2_adam_le_net_fcb_n3_batch_normalization_1_beta_v_read_readvariableopKsavev2_adam_le_net_fcb_n3_batch_normalization_2_gamma_v_read_readvariableopJsavev2_adam_le_net_fcb_n3_batch_normalization_2_beta_v_read_readvariableop>savev2_adam_le_net_fcb_n3_dense_1_kernel_v_read_readvariableop<savev2_adam_le_net_fcb_n3_dense_1_bias_v_read_readvariableop>savev2_adam_le_net_fcb_n3_dense_2_kernel_v_read_readvariableop<savev2_adam_le_net_fcb_n3_dense_2_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *T
dtypesJ
H2F	2
SaveV2º
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes¡
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*½
_input_shapes«
¨: : : : : : : : @:@:@::
ð:ð:ð:ð:ð:ð:T:T:T:T:	ðT:T:T:: : : : : : : : : : : : : : @:@:@::
ð:ð:ð:ð:T:T:	ðT:T:T:: : : : : @:@:@::
ð:ð:ð:ð:T:T:	ðT:T:T:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
: @: 

_output_shapes
:@:-	)
'
_output_shapes
:@:!


_output_shapes	
::&"
 
_output_shapes
:
ð:!

_output_shapes	
:ð:!

_output_shapes	
:ð:!

_output_shapes	
:ð:!

_output_shapes	
:ð:!

_output_shapes	
:ð: 

_output_shapes
:T: 

_output_shapes
:T: 

_output_shapes
:T: 

_output_shapes
:T:%!

_output_shapes
:	ðT: 

_output_shapes
:T:$ 

_output_shapes

:T: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
: :!

_output_shapes
: :,"(
&
_output_shapes
: : #

_output_shapes
: : $

_output_shapes
: : %

_output_shapes
: :,&(
&
_output_shapes
: @: '

_output_shapes
:@:-()
'
_output_shapes
:@:!)

_output_shapes	
::&*"
 
_output_shapes
:
ð:!+

_output_shapes	
:ð:!,

_output_shapes	
:ð:!-

_output_shapes	
:ð: .

_output_shapes
:T: /

_output_shapes
:T:%0!

_output_shapes
:	ðT: 1

_output_shapes
:T:$2 

_output_shapes

:T: 3

_output_shapes
::,4(
&
_output_shapes
: : 5

_output_shapes
: : 6

_output_shapes
: : 7

_output_shapes
: :,8(
&
_output_shapes
: @: 9

_output_shapes
:@:-:)
'
_output_shapes
:@:!;

_output_shapes	
::&<"
 
_output_shapes
:
ð:!=

_output_shapes	
:ð:!>

_output_shapes	
:ð:!?

_output_shapes	
:ð: @

_output_shapes
:T: A

_output_shapes
:T:%B!

_output_shapes
:	ðT: C

_output_shapes
:T:$D 

_output_shapes

:T: E

_output_shapes
::F

_output_shapes
: 
È
N
2__inference_spatial_dropout2d_1_layer_call_fn_6561

inputs
identityÔ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_spatial_dropout2d_1_layer_call_and_return_conditional_losses_53032
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ô
k
2__inference_spatial_dropout2d_1_layer_call_fn_6556

inputs
identity¢StatefulPartitionedCallì
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_spatial_dropout2d_1_layer_call_and_return_conditional_losses_52982
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
®
©
A__inference_dense_2_layer_call_and_return_conditional_losses_6814

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:T*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Softmaxe
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿT:::O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT
 
_user_specified_nameinputs
¥
l
M__inference_spatial_dropout2d_1_layer_call_and_return_conditional_losses_6508

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ì
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/Const
dropout/MulMulinputsdropout/Const:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dropout/Mul
dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2 
dropout/random_uniform/shape/1
dropout/random_uniform/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2 
dropout/random_uniform/shape/2ö
dropout/random_uniform/shapePackstrided_slice:output:0'dropout/random_uniform/shape/1:output:0'dropout/random_uniform/shape/2:output:0strided_slice_1:output:0*
N*
T0*
_output_shapes
:2
dropout/random_uniform/shapeÔ
$dropout/random_uniform/RandomUniformRandomUniform%dropout/random_uniform/shape:output:0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2
dropout/GreaterEqual/yÏ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dropout/Mul_1
IdentityIdentitydropout/Mul_1:z:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ì
i
0__inference_spatial_dropout2d_layer_call_fn_6422

inputs
identity¢StatefulPartitionedCallé
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_spatial_dropout2d_layer_call_and_return_conditional_losses_52322
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
¡
l
M__inference_spatial_dropout2d_1_layer_call_and_return_conditional_losses_5298

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ì
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/Const|
dropout/MulMulinputsdropout/Const:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/Mul
dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2 
dropout/random_uniform/shape/1
dropout/random_uniform/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2 
dropout/random_uniform/shape/2ö
dropout/random_uniform/shapePackstrided_slice:output:0'dropout/random_uniform/shape/1:output:0'dropout/random_uniform/shape/2:output:0strided_slice_1:output:0*
N*
T0*
_output_shapes
:2
dropout/random_uniform/shapeÔ
$dropout/random_uniform/RandomUniformRandomUniform%dropout/random_uniform/shape:output:0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2
dropout/GreaterEqual/yÏ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/Mul_1n
IdentityIdentitydropout/Mul_1:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ô
k
M__inference_spatial_dropout2d_1_layer_call_and_return_conditional_losses_5303

inputs

identity_1c
IdentityIdentityinputs*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityr

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1"!

identity_1Identity_1:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ã(
¸
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_6737

inputs
assignmovingavg_6712
assignmovingavg_1_6718 
cast_readvariableop_resource"
cast_1_readvariableop_resource
identity¢#AssignMovingAvg/AssignSubVariableOp¢%AssignMovingAvg_1/AssignSubVariableOp
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indices
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:T*
	keep_dims(2
moments/mean|
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:T2
moments/StopGradient¤
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT2
moments/SquaredDifference
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices²
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:T*
	keep_dims(2
moments/variance
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:T*
squeeze_dims
 2
moments/Squeeze
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:T*
squeeze_dims
 2
moments/Squeeze_1
AssignMovingAvg/decayConst*'
_class
loc:@AssignMovingAvg/6712*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg/decay
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_6712*
_output_shapes
:T*
dtype02 
AssignMovingAvg/ReadVariableOpÁ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*'
_class
loc:@AssignMovingAvg/6712*
_output_shapes
:T2
AssignMovingAvg/sub¸
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*'
_class
loc:@AssignMovingAvg/6712*
_output_shapes
:T2
AssignMovingAvg/mulý
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_6712AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*'
_class
loc:@AssignMovingAvg/6712*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp¢
AssignMovingAvg_1/decayConst*)
_class
loc:@AssignMovingAvg_1/6718*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg_1/decay
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_6718*
_output_shapes
:T*
dtype02"
 AssignMovingAvg_1/ReadVariableOpË
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*)
_class
loc:@AssignMovingAvg_1/6718*
_output_shapes
:T2
AssignMovingAvg_1/subÂ
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*)
_class
loc:@AssignMovingAvg_1/6718*
_output_shapes
:T2
AssignMovingAvg_1/mul
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_6718AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*)
_class
loc:@AssignMovingAvg_1/6718*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:T*
dtype02
Cast/ReadVariableOp
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
:T*
dtype02
Cast_1/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:T2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:T2
batchnorm/Rsqrt~
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes
:T2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:T2
batchnorm/mul_2|
batchnorm/subSubCast/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:T2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT2
batchnorm/add_1µ
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿT::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT
 
_user_specified_nameinputs
¥
l
M__inference_spatial_dropout2d_1_layer_call_and_return_conditional_losses_4772

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ì
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/Const
dropout/MulMulinputsdropout/Const:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dropout/Mul
dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2 
dropout/random_uniform/shape/1
dropout/random_uniform/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2 
dropout/random_uniform/shape/2ö
dropout/random_uniform/shapePackstrided_slice:output:0'dropout/random_uniform/shape/1:output:0'dropout/random_uniform/shape/2:output:0strided_slice_1:output:0*
N*
T0*
_output_shapes
:2
dropout/random_uniform/shapeÔ
$dropout/random_uniform/RandomUniformRandomUniform%dropout/random_uniform/shape:output:0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2
dropout/GreaterEqual/yÏ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dropout/Mul_1
IdentityIdentitydropout/Mul_1:z:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
­
§
?__inference_dense_layer_call_and_return_conditional_losses_5343

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
ð*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿð2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ð*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿð2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿð2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿð2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:::P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
È
_
A__inference_dropout_layer_call_and_return_conditional_losses_5376

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿð2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿð2

Identity_1"!

identity_1Identity_1:output:0*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿð:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿð
 
_user_specified_nameinputs
Î
Ø
,__inference_le_net_fcb_n3_layer_call_fn_6221

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22
identity¢StatefulPartitionedCall¢
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22*$
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*:
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_le_net_fcb_n3_layer_call_and_return_conditional_losses_56632
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*
_input_shapes}
{:ÿÿÿÿÿÿÿÿÿ  ::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
 
_user_specified_nameinputs
£
j
K__inference_spatial_dropout2d_layer_call_and_return_conditional_losses_6450

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ì
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/Const
dropout/MulMulinputsdropout/Const:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dropout/Mul
dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2 
dropout/random_uniform/shape/1
dropout/random_uniform/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2 
dropout/random_uniform/shape/2ö
dropout/random_uniform/shapePackstrided_slice:output:0'dropout/random_uniform/shape/1:output:0'dropout/random_uniform/shape/2:output:0strided_slice_1:output:0*
N*
T0*
_output_shapes
:2
dropout/random_uniform/shapeÔ
$dropout/random_uniform/RandomUniformRandomUniform%dropout/random_uniform/shape:output:0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2
dropout/GreaterEqual/yÏ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dropout/Mul_1
IdentityIdentitydropout/Mul_1:z:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¡
l
M__inference_spatial_dropout2d_1_layer_call_and_return_conditional_losses_6546

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ì
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/Const|
dropout/MulMulinputsdropout/Const:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/Mul
dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2 
dropout/random_uniform/shape/1
dropout/random_uniform/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2 
dropout/random_uniform/shape/2ö
dropout/random_uniform/shapePackstrided_slice:output:0'dropout/random_uniform/shape/1:output:0'dropout/random_uniform/shape/2:output:0strided_slice_1:output:0*
N*
T0*
_output_shapes
:2
dropout/random_uniform/shapeÔ
$dropout/random_uniform/RandomUniformRandomUniform%dropout/random_uniform/shape:output:0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2
dropout/GreaterEqual/yÏ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/Mul_1n
IdentityIdentitydropout/Mul_1:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
	
ª
B__inference_conv2d_1_layer_call_and_return_conditional_losses_6380

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
Conv2D/ReadVariableOp¤
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ :::W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs

B
&__inference_dropout_layer_call_fn_6619

inputs
identityÀ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿð* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_53762
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿð2

Identity"
identityIdentity:output:0*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿð:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿð
 
_user_specified_nameinputs
ªª
£,
 __inference__traced_restore_7270
file_prefix0
,assignvariableop_le_net_fcb_n3_conv2d_kernel0
,assignvariableop_1_le_net_fcb_n3_conv2d_bias>
:assignvariableop_2_le_net_fcb_n3_batch_normalization_gamma=
9assignvariableop_3_le_net_fcb_n3_batch_normalization_betaD
@assignvariableop_4_le_net_fcb_n3_batch_normalization_moving_meanH
Dassignvariableop_5_le_net_fcb_n3_batch_normalization_moving_variance4
0assignvariableop_6_le_net_fcb_n3_conv2d_1_kernel2
.assignvariableop_7_le_net_fcb_n3_conv2d_1_bias4
0assignvariableop_8_le_net_fcb_n3_conv2d_2_kernel2
.assignvariableop_9_le_net_fcb_n3_conv2d_2_bias2
.assignvariableop_10_le_net_fcb_n3_dense_kernel0
,assignvariableop_11_le_net_fcb_n3_dense_biasA
=assignvariableop_12_le_net_fcb_n3_batch_normalization_1_gamma@
<assignvariableop_13_le_net_fcb_n3_batch_normalization_1_betaG
Cassignvariableop_14_le_net_fcb_n3_batch_normalization_1_moving_meanK
Gassignvariableop_15_le_net_fcb_n3_batch_normalization_1_moving_varianceA
=assignvariableop_16_le_net_fcb_n3_batch_normalization_2_gamma@
<assignvariableop_17_le_net_fcb_n3_batch_normalization_2_betaG
Cassignvariableop_18_le_net_fcb_n3_batch_normalization_2_moving_meanK
Gassignvariableop_19_le_net_fcb_n3_batch_normalization_2_moving_variance4
0assignvariableop_20_le_net_fcb_n3_dense_1_kernel2
.assignvariableop_21_le_net_fcb_n3_dense_1_bias4
0assignvariableop_22_le_net_fcb_n3_dense_2_kernel2
.assignvariableop_23_le_net_fcb_n3_dense_2_bias!
assignvariableop_24_adam_iter#
assignvariableop_25_adam_beta_1#
assignvariableop_26_adam_beta_2"
assignvariableop_27_adam_decay*
&assignvariableop_28_adam_learning_rate
assignvariableop_29_total
assignvariableop_30_count
assignvariableop_31_total_1
assignvariableop_32_count_1:
6assignvariableop_33_adam_le_net_fcb_n3_conv2d_kernel_m8
4assignvariableop_34_adam_le_net_fcb_n3_conv2d_bias_mF
Bassignvariableop_35_adam_le_net_fcb_n3_batch_normalization_gamma_mE
Aassignvariableop_36_adam_le_net_fcb_n3_batch_normalization_beta_m<
8assignvariableop_37_adam_le_net_fcb_n3_conv2d_1_kernel_m:
6assignvariableop_38_adam_le_net_fcb_n3_conv2d_1_bias_m<
8assignvariableop_39_adam_le_net_fcb_n3_conv2d_2_kernel_m:
6assignvariableop_40_adam_le_net_fcb_n3_conv2d_2_bias_m9
5assignvariableop_41_adam_le_net_fcb_n3_dense_kernel_m7
3assignvariableop_42_adam_le_net_fcb_n3_dense_bias_mH
Dassignvariableop_43_adam_le_net_fcb_n3_batch_normalization_1_gamma_mG
Cassignvariableop_44_adam_le_net_fcb_n3_batch_normalization_1_beta_mH
Dassignvariableop_45_adam_le_net_fcb_n3_batch_normalization_2_gamma_mG
Cassignvariableop_46_adam_le_net_fcb_n3_batch_normalization_2_beta_m;
7assignvariableop_47_adam_le_net_fcb_n3_dense_1_kernel_m9
5assignvariableop_48_adam_le_net_fcb_n3_dense_1_bias_m;
7assignvariableop_49_adam_le_net_fcb_n3_dense_2_kernel_m9
5assignvariableop_50_adam_le_net_fcb_n3_dense_2_bias_m:
6assignvariableop_51_adam_le_net_fcb_n3_conv2d_kernel_v8
4assignvariableop_52_adam_le_net_fcb_n3_conv2d_bias_vF
Bassignvariableop_53_adam_le_net_fcb_n3_batch_normalization_gamma_vE
Aassignvariableop_54_adam_le_net_fcb_n3_batch_normalization_beta_v<
8assignvariableop_55_adam_le_net_fcb_n3_conv2d_1_kernel_v:
6assignvariableop_56_adam_le_net_fcb_n3_conv2d_1_bias_v<
8assignvariableop_57_adam_le_net_fcb_n3_conv2d_2_kernel_v:
6assignvariableop_58_adam_le_net_fcb_n3_conv2d_2_bias_v9
5assignvariableop_59_adam_le_net_fcb_n3_dense_kernel_v7
3assignvariableop_60_adam_le_net_fcb_n3_dense_bias_vH
Dassignvariableop_61_adam_le_net_fcb_n3_batch_normalization_1_gamma_vG
Cassignvariableop_62_adam_le_net_fcb_n3_batch_normalization_1_beta_vH
Dassignvariableop_63_adam_le_net_fcb_n3_batch_normalization_2_gamma_vG
Cassignvariableop_64_adam_le_net_fcb_n3_batch_normalization_2_beta_v;
7assignvariableop_65_adam_le_net_fcb_n3_dense_1_kernel_v9
5assignvariableop_66_adam_le_net_fcb_n3_dense_1_bias_v;
7assignvariableop_67_adam_le_net_fcb_n3_dense_2_kernel_v9
5assignvariableop_68_adam_le_net_fcb_n3_dense_2_bias_v
identity_70¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_27¢AssignVariableOp_28¢AssignVariableOp_29¢AssignVariableOp_3¢AssignVariableOp_30¢AssignVariableOp_31¢AssignVariableOp_32¢AssignVariableOp_33¢AssignVariableOp_34¢AssignVariableOp_35¢AssignVariableOp_36¢AssignVariableOp_37¢AssignVariableOp_38¢AssignVariableOp_39¢AssignVariableOp_4¢AssignVariableOp_40¢AssignVariableOp_41¢AssignVariableOp_42¢AssignVariableOp_43¢AssignVariableOp_44¢AssignVariableOp_45¢AssignVariableOp_46¢AssignVariableOp_47¢AssignVariableOp_48¢AssignVariableOp_49¢AssignVariableOp_5¢AssignVariableOp_50¢AssignVariableOp_51¢AssignVariableOp_52¢AssignVariableOp_53¢AssignVariableOp_54¢AssignVariableOp_55¢AssignVariableOp_56¢AssignVariableOp_57¢AssignVariableOp_58¢AssignVariableOp_59¢AssignVariableOp_6¢AssignVariableOp_60¢AssignVariableOp_61¢AssignVariableOp_62¢AssignVariableOp_63¢AssignVariableOp_64¢AssignVariableOp_65¢AssignVariableOp_66¢AssignVariableOp_67¢AssignVariableOp_68¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9í
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:F*
dtype0*ù
valueïBìFB$c1/kernel/.ATTRIBUTES/VARIABLE_VALUEB"c1/bias/.ATTRIBUTES/VARIABLE_VALUEB(affine1/gamma/.ATTRIBUTES/VARIABLE_VALUEB'affine1/beta/.ATTRIBUTES/VARIABLE_VALUEB.affine1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB2affine1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB$c3/kernel/.ATTRIBUTES/VARIABLE_VALUEB"c3/bias/.ATTRIBUTES/VARIABLE_VALUEB'c_add/kernel/.ATTRIBUTES/VARIABLE_VALUEB%c_add/bias/.ATTRIBUTES/VARIABLE_VALUEB$c5/kernel/.ATTRIBUTES/VARIABLE_VALUEB"c5/bias/.ATTRIBUTES/VARIABLE_VALUEB(affine3/gamma/.ATTRIBUTES/VARIABLE_VALUEB'affine3/beta/.ATTRIBUTES/VARIABLE_VALUEB.affine3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB2affine3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB(affine4/gamma/.ATTRIBUTES/VARIABLE_VALUEB'affine4/beta/.ATTRIBUTES/VARIABLE_VALUEB.affine4/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB2affine4/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB$f6/kernel/.ATTRIBUTES/VARIABLE_VALUEB"f6/bias/.ATTRIBUTES/VARIABLE_VALUEB.output_layer/kernel/.ATTRIBUTES/VARIABLE_VALUEB,output_layer/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB@c1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB>c1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBDaffine1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCaffine1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB@c3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB>c3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCc_add/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBAc_add/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB@c5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB>c5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBDaffine3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCaffine3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBDaffine4/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCaffine4/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB@f6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB>f6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBJoutput_layer/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBHoutput_layer/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB@c1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB>c1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBDaffine1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCaffine1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB@c3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB>c3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCc_add/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBAc_add/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB@c5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB>c5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBDaffine3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCaffine3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBDaffine4/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCaffine4/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB@f6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB>f6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBJoutput_layer/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBHoutput_layer/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:F*
dtype0*¡
valueBFB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*®
_output_shapes
::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*T
dtypesJ
H2F	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity«
AssignVariableOpAssignVariableOp,assignvariableop_le_net_fcb_n3_conv2d_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1±
AssignVariableOp_1AssignVariableOp,assignvariableop_1_le_net_fcb_n3_conv2d_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2¿
AssignVariableOp_2AssignVariableOp:assignvariableop_2_le_net_fcb_n3_batch_normalization_gammaIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3¾
AssignVariableOp_3AssignVariableOp9assignvariableop_3_le_net_fcb_n3_batch_normalization_betaIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4Å
AssignVariableOp_4AssignVariableOp@assignvariableop_4_le_net_fcb_n3_batch_normalization_moving_meanIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5É
AssignVariableOp_5AssignVariableOpDassignvariableop_5_le_net_fcb_n3_batch_normalization_moving_varianceIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6µ
AssignVariableOp_6AssignVariableOp0assignvariableop_6_le_net_fcb_n3_conv2d_1_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7³
AssignVariableOp_7AssignVariableOp.assignvariableop_7_le_net_fcb_n3_conv2d_1_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8µ
AssignVariableOp_8AssignVariableOp0assignvariableop_8_le_net_fcb_n3_conv2d_2_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9³
AssignVariableOp_9AssignVariableOp.assignvariableop_9_le_net_fcb_n3_conv2d_2_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10¶
AssignVariableOp_10AssignVariableOp.assignvariableop_10_le_net_fcb_n3_dense_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11´
AssignVariableOp_11AssignVariableOp,assignvariableop_11_le_net_fcb_n3_dense_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12Å
AssignVariableOp_12AssignVariableOp=assignvariableop_12_le_net_fcb_n3_batch_normalization_1_gammaIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13Ä
AssignVariableOp_13AssignVariableOp<assignvariableop_13_le_net_fcb_n3_batch_normalization_1_betaIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14Ë
AssignVariableOp_14AssignVariableOpCassignvariableop_14_le_net_fcb_n3_batch_normalization_1_moving_meanIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15Ï
AssignVariableOp_15AssignVariableOpGassignvariableop_15_le_net_fcb_n3_batch_normalization_1_moving_varianceIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16Å
AssignVariableOp_16AssignVariableOp=assignvariableop_16_le_net_fcb_n3_batch_normalization_2_gammaIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17Ä
AssignVariableOp_17AssignVariableOp<assignvariableop_17_le_net_fcb_n3_batch_normalization_2_betaIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18Ë
AssignVariableOp_18AssignVariableOpCassignvariableop_18_le_net_fcb_n3_batch_normalization_2_moving_meanIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19Ï
AssignVariableOp_19AssignVariableOpGassignvariableop_19_le_net_fcb_n3_batch_normalization_2_moving_varianceIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20¸
AssignVariableOp_20AssignVariableOp0assignvariableop_20_le_net_fcb_n3_dense_1_kernelIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21¶
AssignVariableOp_21AssignVariableOp.assignvariableop_21_le_net_fcb_n3_dense_1_biasIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22¸
AssignVariableOp_22AssignVariableOp0assignvariableop_22_le_net_fcb_n3_dense_2_kernelIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23¶
AssignVariableOp_23AssignVariableOp.assignvariableop_23_le_net_fcb_n3_dense_2_biasIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_24¥
AssignVariableOp_24AssignVariableOpassignvariableop_24_adam_iterIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25§
AssignVariableOp_25AssignVariableOpassignvariableop_25_adam_beta_1Identity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26§
AssignVariableOp_26AssignVariableOpassignvariableop_26_adam_beta_2Identity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27¦
AssignVariableOp_27AssignVariableOpassignvariableop_27_adam_decayIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28®
AssignVariableOp_28AssignVariableOp&assignvariableop_28_adam_learning_rateIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29¡
AssignVariableOp_29AssignVariableOpassignvariableop_29_totalIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30¡
AssignVariableOp_30AssignVariableOpassignvariableop_30_countIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31£
AssignVariableOp_31AssignVariableOpassignvariableop_31_total_1Identity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32£
AssignVariableOp_32AssignVariableOpassignvariableop_32_count_1Identity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33¾
AssignVariableOp_33AssignVariableOp6assignvariableop_33_adam_le_net_fcb_n3_conv2d_kernel_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34¼
AssignVariableOp_34AssignVariableOp4assignvariableop_34_adam_le_net_fcb_n3_conv2d_bias_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35Ê
AssignVariableOp_35AssignVariableOpBassignvariableop_35_adam_le_net_fcb_n3_batch_normalization_gamma_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36É
AssignVariableOp_36AssignVariableOpAassignvariableop_36_adam_le_net_fcb_n3_batch_normalization_beta_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37À
AssignVariableOp_37AssignVariableOp8assignvariableop_37_adam_le_net_fcb_n3_conv2d_1_kernel_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38¾
AssignVariableOp_38AssignVariableOp6assignvariableop_38_adam_le_net_fcb_n3_conv2d_1_bias_mIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39À
AssignVariableOp_39AssignVariableOp8assignvariableop_39_adam_le_net_fcb_n3_conv2d_2_kernel_mIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40¾
AssignVariableOp_40AssignVariableOp6assignvariableop_40_adam_le_net_fcb_n3_conv2d_2_bias_mIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41½
AssignVariableOp_41AssignVariableOp5assignvariableop_41_adam_le_net_fcb_n3_dense_kernel_mIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42»
AssignVariableOp_42AssignVariableOp3assignvariableop_42_adam_le_net_fcb_n3_dense_bias_mIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43Ì
AssignVariableOp_43AssignVariableOpDassignvariableop_43_adam_le_net_fcb_n3_batch_normalization_1_gamma_mIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44Ë
AssignVariableOp_44AssignVariableOpCassignvariableop_44_adam_le_net_fcb_n3_batch_normalization_1_beta_mIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45Ì
AssignVariableOp_45AssignVariableOpDassignvariableop_45_adam_le_net_fcb_n3_batch_normalization_2_gamma_mIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46Ë
AssignVariableOp_46AssignVariableOpCassignvariableop_46_adam_le_net_fcb_n3_batch_normalization_2_beta_mIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47¿
AssignVariableOp_47AssignVariableOp7assignvariableop_47_adam_le_net_fcb_n3_dense_1_kernel_mIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48½
AssignVariableOp_48AssignVariableOp5assignvariableop_48_adam_le_net_fcb_n3_dense_1_bias_mIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49¿
AssignVariableOp_49AssignVariableOp7assignvariableop_49_adam_le_net_fcb_n3_dense_2_kernel_mIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50½
AssignVariableOp_50AssignVariableOp5assignvariableop_50_adam_le_net_fcb_n3_dense_2_bias_mIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51¾
AssignVariableOp_51AssignVariableOp6assignvariableop_51_adam_le_net_fcb_n3_conv2d_kernel_vIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52¼
AssignVariableOp_52AssignVariableOp4assignvariableop_52_adam_le_net_fcb_n3_conv2d_bias_vIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53Ê
AssignVariableOp_53AssignVariableOpBassignvariableop_53_adam_le_net_fcb_n3_batch_normalization_gamma_vIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54É
AssignVariableOp_54AssignVariableOpAassignvariableop_54_adam_le_net_fcb_n3_batch_normalization_beta_vIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_54n
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:2
Identity_55À
AssignVariableOp_55AssignVariableOp8assignvariableop_55_adam_le_net_fcb_n3_conv2d_1_kernel_vIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_55n
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:2
Identity_56¾
AssignVariableOp_56AssignVariableOp6assignvariableop_56_adam_le_net_fcb_n3_conv2d_1_bias_vIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_56n
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:2
Identity_57À
AssignVariableOp_57AssignVariableOp8assignvariableop_57_adam_le_net_fcb_n3_conv2d_2_kernel_vIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_57n
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:2
Identity_58¾
AssignVariableOp_58AssignVariableOp6assignvariableop_58_adam_le_net_fcb_n3_conv2d_2_bias_vIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_58n
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:2
Identity_59½
AssignVariableOp_59AssignVariableOp5assignvariableop_59_adam_le_net_fcb_n3_dense_kernel_vIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59n
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:2
Identity_60»
AssignVariableOp_60AssignVariableOp3assignvariableop_60_adam_le_net_fcb_n3_dense_bias_vIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_60n
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:2
Identity_61Ì
AssignVariableOp_61AssignVariableOpDassignvariableop_61_adam_le_net_fcb_n3_batch_normalization_1_gamma_vIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_61n
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:2
Identity_62Ë
AssignVariableOp_62AssignVariableOpCassignvariableop_62_adam_le_net_fcb_n3_batch_normalization_1_beta_vIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_62n
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:2
Identity_63Ì
AssignVariableOp_63AssignVariableOpDassignvariableop_63_adam_le_net_fcb_n3_batch_normalization_2_gamma_vIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_63n
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:2
Identity_64Ë
AssignVariableOp_64AssignVariableOpCassignvariableop_64_adam_le_net_fcb_n3_batch_normalization_2_beta_vIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_64n
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:2
Identity_65¿
AssignVariableOp_65AssignVariableOp7assignvariableop_65_adam_le_net_fcb_n3_dense_1_kernel_vIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_65n
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:2
Identity_66½
AssignVariableOp_66AssignVariableOp5assignvariableop_66_adam_le_net_fcb_n3_dense_1_bias_vIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_66n
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:2
Identity_67¿
AssignVariableOp_67AssignVariableOp7assignvariableop_67_adam_le_net_fcb_n3_dense_2_kernel_vIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_67n
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:2
Identity_68½
AssignVariableOp_68AssignVariableOp5assignvariableop_68_adam_le_net_fcb_n3_dense_2_bias_vIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_689
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpÌ
Identity_69Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_69¿
Identity_70IdentityIdentity_69:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_70"#
identity_70Identity_70:output:0*«
_input_shapes
: :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
	
¨
@__inference_conv2d_layer_call_and_return_conditional_losses_5092

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp¤
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ  :::W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
 
_user_specified_nameinputs
ô
k
M__inference_spatial_dropout2d_1_layer_call_and_return_conditional_losses_6551

inputs

identity_1c
IdentityIdentityinputs*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityr

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1"!

identity_1Identity_1:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

¥
2__inference_batch_normalization_layer_call_fn_6356

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall¯
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_batch_normalization_layer_call_and_return_conditional_losses_45832
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
Ñ
Ù
,__inference_le_net_fcb_n3_layer_call_fn_5767
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22
identity¢StatefulPartitionedCall£
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22*$
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*:
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_le_net_fcb_n3_layer_call_and_return_conditional_losses_56632
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*
_input_shapes}
{:ÿÿÿÿÿÿÿÿÿ  ::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
!
_user_specified_name	input_1
ý
c
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_4631

inputs
identity­
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
2	
MaxPool
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Å
ª
M__inference_batch_normalization_layer_call_and_return_conditional_losses_4583

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ê
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3ÿ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1¦
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
×q
°

G__inference_le_net_fcb_n3_layer_call_and_return_conditional_losses_6115

inputs)
%conv2d_conv2d_readvariableop_resource*
&conv2d_biasadd_readvariableop_resource/
+batch_normalization_readvariableop_resource1
-batch_normalization_readvariableop_1_resource@
<batch_normalization_fusedbatchnormv3_readvariableop_resourceB
>batch_normalization_fusedbatchnormv3_readvariableop_1_resource+
'conv2d_1_conv2d_readvariableop_resource,
(conv2d_1_biasadd_readvariableop_resource+
'conv2d_2_conv2d_readvariableop_resource,
(conv2d_2_biasadd_readvariableop_resource(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource6
2batch_normalization_1_cast_readvariableop_resource8
4batch_normalization_1_cast_1_readvariableop_resource8
4batch_normalization_1_cast_2_readvariableop_resource8
4batch_normalization_1_cast_3_readvariableop_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource6
2batch_normalization_2_cast_readvariableop_resource8
4batch_normalization_2_cast_1_readvariableop_resource8
4batch_normalization_2_cast_2_readvariableop_resource8
4batch_normalization_2_cast_3_readvariableop_resource*
&dense_2_matmul_readvariableop_resource+
'dense_2_biasadd_readvariableop_resource
identityª
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
conv2d/Conv2D/ReadVariableOp¹
conv2d/Conv2DConv2Dinputs$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingVALID*
strides
2
conv2d/Conv2D¡
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
conv2d/BiasAdd/ReadVariableOp¤
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
conv2d/BiasAddu
conv2d/ReluReluconv2d/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
conv2d/Relu°
"batch_normalization/ReadVariableOpReadVariableOp+batch_normalization_readvariableop_resource*
_output_shapes
: *
dtype02$
"batch_normalization/ReadVariableOp¶
$batch_normalization/ReadVariableOp_1ReadVariableOp-batch_normalization_readvariableop_1_resource*
_output_shapes
: *
dtype02&
$batch_normalization/ReadVariableOp_1ã
3batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype025
3batch_normalization/FusedBatchNormV3/ReadVariableOpé
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype027
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1Õ
$batch_normalization/FusedBatchNormV3FusedBatchNormV3conv2d/Relu:activations:0*batch_normalization/ReadVariableOp:value:0,batch_normalization/ReadVariableOp_1:value:0;batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0=batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
is_training( 2&
$batch_normalization/FusedBatchNormV3x
ReluRelu(batch_normalization/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Reluº
max_pooling2d/MaxPoolMaxPoolRelu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
ksize
*
paddingVALID*
strides
2
max_pooling2d/MaxPool°
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02 
conv2d_1/Conv2D/ReadVariableOp×
conv2d_1/Conv2DConv2Dmax_pooling2d/MaxPool:output:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
2
conv2d_1/Conv2D§
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_1/BiasAdd/ReadVariableOp¬
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
conv2d_1/BiasAdd{
conv2d_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
conv2d_1/Relu
spatial_dropout2d/IdentityIdentityconv2d_1/Relu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
spatial_dropout2d/Identity±
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02 
conv2d_2/Conv2D/ReadVariableOpÝ
conv2d_2/Conv2DConv2D#spatial_dropout2d/Identity:output:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2
conv2d_2/Conv2D¨
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02!
conv2d_2/BiasAdd/ReadVariableOp­
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv2d_2/BiasAdd|
conv2d_2/ReluReluconv2d_2/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv2d_2/Relu 
spatial_dropout2d_1/IdentityIdentityconv2d_2/Relu:activations:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
spatial_dropout2d_1/IdentityÒ
max_pooling2d_2/MaxPoolMaxPool%spatial_dropout2d_1/Identity:output:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
2
max_pooling2d_2/MaxPoolu
Relu_1Relu max_pooling2d_2/MaxPool:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu_1Á
max_pooling2d_1/MaxPoolMaxPoolRelu_1:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
2
max_pooling2d_1/MaxPoolo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
flatten/Const
flatten/ReshapeReshape max_pooling2d_1/MaxPool:output:0flatten/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
flatten/Reshape¡
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
ð*
dtype02
dense/MatMul/ReadVariableOp
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿð2
dense/MatMul
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:ð*
dtype02
dense/BiasAdd/ReadVariableOp
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿð2
dense/BiasAddk

dense/ReluReludense/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿð2

dense/Relu}
dropout/IdentityIdentitydense/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿð2
dropout/IdentityÆ
)batch_normalization_1/Cast/ReadVariableOpReadVariableOp2batch_normalization_1_cast_readvariableop_resource*
_output_shapes	
:ð*
dtype02+
)batch_normalization_1/Cast/ReadVariableOpÌ
+batch_normalization_1/Cast_1/ReadVariableOpReadVariableOp4batch_normalization_1_cast_1_readvariableop_resource*
_output_shapes	
:ð*
dtype02-
+batch_normalization_1/Cast_1/ReadVariableOpÌ
+batch_normalization_1/Cast_2/ReadVariableOpReadVariableOp4batch_normalization_1_cast_2_readvariableop_resource*
_output_shapes	
:ð*
dtype02-
+batch_normalization_1/Cast_2/ReadVariableOpÌ
+batch_normalization_1/Cast_3/ReadVariableOpReadVariableOp4batch_normalization_1_cast_3_readvariableop_resource*
_output_shapes	
:ð*
dtype02-
+batch_normalization_1/Cast_3/ReadVariableOp
%batch_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2'
%batch_normalization_1/batchnorm/add/yÞ
#batch_normalization_1/batchnorm/addAddV23batch_normalization_1/Cast_1/ReadVariableOp:value:0.batch_normalization_1/batchnorm/add/y:output:0*
T0*
_output_shapes	
:ð2%
#batch_normalization_1/batchnorm/add¦
%batch_normalization_1/batchnorm/RsqrtRsqrt'batch_normalization_1/batchnorm/add:z:0*
T0*
_output_shapes	
:ð2'
%batch_normalization_1/batchnorm/Rsqrt×
#batch_normalization_1/batchnorm/mulMul)batch_normalization_1/batchnorm/Rsqrt:y:03batch_normalization_1/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:ð2%
#batch_normalization_1/batchnorm/mulÌ
%batch_normalization_1/batchnorm/mul_1Muldropout/Identity:output:0'batch_normalization_1/batchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿð2'
%batch_normalization_1/batchnorm/mul_1×
%batch_normalization_1/batchnorm/mul_2Mul1batch_normalization_1/Cast/ReadVariableOp:value:0'batch_normalization_1/batchnorm/mul:z:0*
T0*
_output_shapes	
:ð2'
%batch_normalization_1/batchnorm/mul_2×
#batch_normalization_1/batchnorm/subSub3batch_normalization_1/Cast_2/ReadVariableOp:value:0)batch_normalization_1/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:ð2%
#batch_normalization_1/batchnorm/subÞ
%batch_normalization_1/batchnorm/add_1AddV2)batch_normalization_1/batchnorm/mul_1:z:0'batch_normalization_1/batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿð2'
%batch_normalization_1/batchnorm/add_1v
Relu_2Relu)batch_normalization_1/batchnorm/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿð2
Relu_2¦
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	ðT*
dtype02
dense_1/MatMul/ReadVariableOp
dense_1/MatMulMatMulRelu_2:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT2
dense_1/MatMul¤
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:T*
dtype02 
dense_1/BiasAdd/ReadVariableOp¡
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT2
dense_1/BiasAddp
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT2
dense_1/ReluÅ
)batch_normalization_2/Cast/ReadVariableOpReadVariableOp2batch_normalization_2_cast_readvariableop_resource*
_output_shapes
:T*
dtype02+
)batch_normalization_2/Cast/ReadVariableOpË
+batch_normalization_2/Cast_1/ReadVariableOpReadVariableOp4batch_normalization_2_cast_1_readvariableop_resource*
_output_shapes
:T*
dtype02-
+batch_normalization_2/Cast_1/ReadVariableOpË
+batch_normalization_2/Cast_2/ReadVariableOpReadVariableOp4batch_normalization_2_cast_2_readvariableop_resource*
_output_shapes
:T*
dtype02-
+batch_normalization_2/Cast_2/ReadVariableOpË
+batch_normalization_2/Cast_3/ReadVariableOpReadVariableOp4batch_normalization_2_cast_3_readvariableop_resource*
_output_shapes
:T*
dtype02-
+batch_normalization_2/Cast_3/ReadVariableOp
%batch_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2'
%batch_normalization_2/batchnorm/add/yÝ
#batch_normalization_2/batchnorm/addAddV23batch_normalization_2/Cast_1/ReadVariableOp:value:0.batch_normalization_2/batchnorm/add/y:output:0*
T0*
_output_shapes
:T2%
#batch_normalization_2/batchnorm/add¥
%batch_normalization_2/batchnorm/RsqrtRsqrt'batch_normalization_2/batchnorm/add:z:0*
T0*
_output_shapes
:T2'
%batch_normalization_2/batchnorm/RsqrtÖ
#batch_normalization_2/batchnorm/mulMul)batch_normalization_2/batchnorm/Rsqrt:y:03batch_normalization_2/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:T2%
#batch_normalization_2/batchnorm/mulÌ
%batch_normalization_2/batchnorm/mul_1Muldense_1/Relu:activations:0'batch_normalization_2/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT2'
%batch_normalization_2/batchnorm/mul_1Ö
%batch_normalization_2/batchnorm/mul_2Mul1batch_normalization_2/Cast/ReadVariableOp:value:0'batch_normalization_2/batchnorm/mul:z:0*
T0*
_output_shapes
:T2'
%batch_normalization_2/batchnorm/mul_2Ö
#batch_normalization_2/batchnorm/subSub3batch_normalization_2/Cast_2/ReadVariableOp:value:0)batch_normalization_2/batchnorm/mul_2:z:0*
T0*
_output_shapes
:T2%
#batch_normalization_2/batchnorm/subÝ
%batch_normalization_2/batchnorm/add_1AddV2)batch_normalization_2/batchnorm/mul_1:z:0'batch_normalization_2/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT2'
%batch_normalization_2/batchnorm/add_1z
SoftmaxSoftmax)batch_normalization_2/batchnorm/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT2	
Softmax¥
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

:T*
dtype02
dense_2/MatMul/ReadVariableOp
dense_2/MatMulMatMulSoftmax:softmax:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_2/MatMul¤
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_2/BiasAdd/ReadVariableOp¡
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_2/BiasAddy
dense_2/SoftmaxSoftmaxdense_2/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_2/Softmaxm
IdentityIdentitydense_2/Softmax:softmax:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*
_input_shapes}
{:ÿÿÿÿÿÿÿÿÿ  :::::::::::::::::::::::::W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
 
_user_specified_nameinputs
Í
¥
2__inference_batch_normalization_layer_call_fn_6292

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_batch_normalization_layer_call_and_return_conditional_losses_51272
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ ::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
 
B
&__inference_flatten_layer_call_fn_6572

inputs
identityÀ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_flatten_layer_call_and_return_conditional_losses_53242
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ý
ª
M__inference_batch_normalization_layer_call_and_return_conditional_losses_5127

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ø
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3ÿ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
Û
i
K__inference_spatial_dropout2d_layer_call_and_return_conditional_losses_4714

inputs

identity_1}
IdentityIdentityinputs*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity

Identity_1IdentityIdentity:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity_1"!

identity_1Identity_1:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
	
ª
B__inference_conv2d_2_layer_call_and_return_conditional_losses_6476

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOp¥
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ@:::W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
ÿ
e
I__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_4791

inputs
identity­
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
2	
MaxPool
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
	
¨
@__inference_conv2d_layer_call_and_return_conditional_losses_6232

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp¤
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ  :::W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
 
_user_specified_nameinputs
±
§
4__inference_batch_normalization_2_layer_call_fn_6770

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_50332
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿT::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT
 
_user_specified_nameinputs
î
i
K__inference_spatial_dropout2d_layer_call_and_return_conditional_losses_5237

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity_1"!

identity_1Identity_1:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
ÿ
e
I__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_4643

inputs
identity­
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
2	
MaxPool
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
»
]
A__inference_flatten_layer_call_and_return_conditional_losses_6567

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

Ï
"__inference_signature_wrapper_5830
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22
identity¢StatefulPartitionedCallû
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22*$
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*:
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *(
f#R!
__inference__wrapped_model_45212
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*
_input_shapes}
{:ÿÿÿÿÿÿÿÿÿ  ::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
!
_user_specified_name	input_1
À
L
0__inference_spatial_dropout2d_layer_call_fn_6427

inputs
identityÑ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_spatial_dropout2d_layer_call_and_return_conditional_losses_52372
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs


M__inference_batch_normalization_layer_call_and_return_conditional_losses_4614

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ü
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ :::::i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
Û(
¸
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_4893

inputs
assignmovingavg_4868
assignmovingavg_1_4874 
cast_readvariableop_resource"
cast_1_readvariableop_resource
identity¢#AssignMovingAvg/AssignSubVariableOp¢%AssignMovingAvg_1/AssignSubVariableOp
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indices
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	ð*
	keep_dims(2
moments/mean}
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	ð2
moments/StopGradient¥
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿð2
moments/SquaredDifference
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices³
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	ð*
	keep_dims(2
moments/variance
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:ð*
squeeze_dims
 2
moments/Squeeze
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:ð*
squeeze_dims
 2
moments/Squeeze_1
AssignMovingAvg/decayConst*'
_class
loc:@AssignMovingAvg/4868*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg/decay
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_4868*
_output_shapes	
:ð*
dtype02 
AssignMovingAvg/ReadVariableOpÂ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*'
_class
loc:@AssignMovingAvg/4868*
_output_shapes	
:ð2
AssignMovingAvg/sub¹
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*'
_class
loc:@AssignMovingAvg/4868*
_output_shapes	
:ð2
AssignMovingAvg/mulý
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_4868AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*'
_class
loc:@AssignMovingAvg/4868*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp¢
AssignMovingAvg_1/decayConst*)
_class
loc:@AssignMovingAvg_1/4874*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg_1/decay
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_4874*
_output_shapes	
:ð*
dtype02"
 AssignMovingAvg_1/ReadVariableOpÌ
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*)
_class
loc:@AssignMovingAvg_1/4874*
_output_shapes	
:ð2
AssignMovingAvg_1/subÃ
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*)
_class
loc:@AssignMovingAvg_1/4874*
_output_shapes	
:ð2
AssignMovingAvg_1/mul
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_4874AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*)
_class
loc:@AssignMovingAvg_1/4874*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes	
:ð*
dtype02
Cast/ReadVariableOp
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes	
:ð*
dtype02
Cast_1/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:ð2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:ð2
batchnorm/Rsqrt
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes	
:ð2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿð2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:ð2
batchnorm/mul_2}
batchnorm/subSubCast/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:ð2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿð2
batchnorm/add_1¶
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿð2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿð::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿð
 
_user_specified_nameinputs
³
§
4__inference_batch_normalization_2_layer_call_fn_6783

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_50662
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿT::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT
 
_user_specified_nameinputs"¸L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*³
serving_default
C
input_18
serving_default_input_1:0ÿÿÿÿÿÿÿÿÿ  <
output_10
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:ª¬
Ü	
c1
affine1
s2
c3
s4
affine2
	c_add

affine_add
		s_add

flatten
c5
c5_drop
affine3
affine4
f6
output_layer
	optimizer
trainable_variables
	variables
regularization_losses
	keras_api

signatures
û_default_save_signature
+ü&call_and_return_all_conditional_losses
ý__call__"Ø
_tf_keras_model¾{"class_name": "LeNetFCBN3", "name": "le_net_fcb_n3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"layer was saved without config": true}, "is_graph_network": false, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "LeNetFCBN3"}, "training_config": {"loss": {"class_name": "BinaryCrossentropy", "config": {"reduction": "auto", "name": "binary_crossentropy", "from_logits": true, "label_smoothing": 0}}, "metrics": ["accuracy"], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0005000000237487257, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
ð


kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
+þ&call_and_return_all_conditional_losses
ÿ__call__"É	
_tf_keras_layer¯	{"class_name": "Conv2D", "name": "conv2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 32, 32, 1]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 32, 32, 1]}, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32, 32, 1]}}
¸	
axis
	gamma
beta
 moving_mean
!moving_variance
"trainable_variables
#	variables
$regularization_losses
%	keras_api
+&call_and_return_all_conditional_losses
__call__"â
_tf_keras_layerÈ{"class_name": "BatchNormalization", "name": "batch_normalization", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 30, 30, 32]}}
ý
&trainable_variables
'	variables
(regularization_losses
)	keras_api
+&call_and_return_all_conditional_losses
__call__"ì
_tf_keras_layerÒ{"class_name": "MaxPooling2D", "name": "max_pooling2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
õ	

*kernel
+bias
,trainable_variables
-	variables
.regularization_losses
/	keras_api
+&call_and_return_all_conditional_losses
__call__"Î
_tf_keras_layer´{"class_name": "Conv2D", "name": "conv2d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 15, 15, 32]}}

0trainable_variables
1	variables
2regularization_losses
3	keras_api
+&call_and_return_all_conditional_losses
__call__"ð
_tf_keras_layerÖ{"class_name": "MaxPooling2D", "name": "max_pooling2d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}

4trainable_variables
5	variables
6regularization_losses
7	keras_api
+&call_and_return_all_conditional_losses
__call__"ÿ
_tf_keras_layerå{"class_name": "SpatialDropout2D", "name": "spatial_dropout2d", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "spatial_dropout2d", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
÷


8kernel
9bias
:trainable_variables
;	variables
<regularization_losses
=	keras_api
+&call_and_return_all_conditional_losses
__call__"Ð	
_tf_keras_layer¶	{"class_name": "Conv2D", "name": "conv2d_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 32, 32, 1]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_2", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 32, 32, 1]}, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 13, 13, 64]}}

>trainable_variables
?	variables
@regularization_losses
A	keras_api
+&call_and_return_all_conditional_losses
__call__"
_tf_keras_layeré{"class_name": "SpatialDropout2D", "name": "spatial_dropout2d_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "spatial_dropout2d_1", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}

Btrainable_variables
C	variables
Dregularization_losses
E	keras_api
+&call_and_return_all_conditional_losses
__call__"ð
_tf_keras_layerÖ{"class_name": "MaxPooling2D", "name": "max_pooling2d_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_2", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
ä
Ftrainable_variables
G	variables
Hregularization_losses
I	keras_api
+&call_and_return_all_conditional_losses
__call__"Ó
_tf_keras_layer¹{"class_name": "Flatten", "name": "flatten", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
ñ

Jkernel
Kbias
Ltrainable_variables
M	variables
Nregularization_losses
O	keras_api
+&call_and_return_all_conditional_losses
__call__"Ê
_tf_keras_layer°{"class_name": "Dense", "name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 240, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 512}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 512]}}
ä
Ptrainable_variables
Q	variables
Rregularization_losses
S	keras_api
+&call_and_return_all_conditional_losses
__call__"Ó
_tf_keras_layer¹{"class_name": "Dropout", "name": "dropout", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.15, "noise_shape": null, "seed": null}}
¶	
Taxis
	Ugamma
Vbeta
Wmoving_mean
Xmoving_variance
Ytrainable_variables
Z	variables
[regularization_losses
\	keras_api
+&call_and_return_all_conditional_losses
__call__"à
_tf_keras_layerÆ{"class_name": "BatchNormalization", "name": "batch_normalization_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_1", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {"1": 240}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 240]}}
´	
]axis
	^gamma
_beta
`moving_mean
amoving_variance
btrainable_variables
c	variables
dregularization_losses
e	keras_api
+&call_and_return_all_conditional_losses
__call__"Þ
_tf_keras_layerÄ{"class_name": "BatchNormalization", "name": "batch_normalization_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_2", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {"1": 84}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 84]}}
ô

fkernel
gbias
htrainable_variables
i	variables
jregularization_losses
k	keras_api
+&call_and_return_all_conditional_losses
__call__"Í
_tf_keras_layer³{"class_name": "Dense", "name": "dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 84, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 240}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 240]}}
ô

lkernel
mbias
ntrainable_variables
o	variables
pregularization_losses
q	keras_api
+&call_and_return_all_conditional_losses
__call__"Í
_tf_keras_layer³{"class_name": "Dense", "name": "dense_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 2, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 84}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 84]}}
»
riter

sbeta_1

tbeta_2
	udecay
vlearning_ratem×mØmÙmÚ*mÛ+mÜ8mÝ9mÞJmßKmàUmáVmâ^mã_mäfmågmælmçmmèvévêvëvì*ví+vî8vï9vðJvñKvòUvóVvô^võ_vöfv÷gvølvùmvú"
	optimizer
¦
0
1
2
3
*4
+5
86
97
J8
K9
U10
V11
^12
_13
f14
g15
l16
m17"
trackable_list_wrapper
Ö
0
1
2
3
 4
!5
*6
+7
88
99
J10
K11
U12
V13
W14
X15
^16
_17
`18
a19
f20
g21
l22
m23"
trackable_list_wrapper
 "
trackable_list_wrapper
Î
wlayer_regularization_losses
xlayer_metrics
trainable_variables
	variables
ynon_trainable_variables
zmetrics
regularization_losses

{layers
ý__call__
û_default_save_signature
+ü&call_and_return_all_conditional_losses
'ü"call_and_return_conditional_losses"
_generic_user_object
-
serving_default"
signature_map
5:3 2le_net_fcb_n3/conv2d/kernel
':% 2le_net_fcb_n3/conv2d/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
±
|layer_regularization_losses
}layer_metrics
trainable_variables
	variables
~non_trainable_variables
metrics
regularization_losses
layers
ÿ__call__
+þ&call_and_return_all_conditional_losses
'þ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
5:3 2'le_net_fcb_n3/batch_normalization/gamma
4:2 2&le_net_fcb_n3/batch_normalization/beta
=:;  (2-le_net_fcb_n3/batch_normalization/moving_mean
A:?  (21le_net_fcb_n3/batch_normalization/moving_variance
.
0
1"
trackable_list_wrapper
<
0
1
 2
!3"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
 layer_regularization_losses
layer_metrics
"trainable_variables
#	variables
non_trainable_variables
metrics
$regularization_losses
layers
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
 layer_regularization_losses
layer_metrics
&trainable_variables
'	variables
non_trainable_variables
metrics
(regularization_losses
layers
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
7:5 @2le_net_fcb_n3/conv2d_1/kernel
):'@2le_net_fcb_n3/conv2d_1/bias
.
*0
+1"
trackable_list_wrapper
.
*0
+1"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
 layer_regularization_losses
layer_metrics
,trainable_variables
-	variables
non_trainable_variables
metrics
.regularization_losses
layers
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
 layer_regularization_losses
layer_metrics
0trainable_variables
1	variables
non_trainable_variables
metrics
2regularization_losses
layers
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
 layer_regularization_losses
layer_metrics
4trainable_variables
5	variables
non_trainable_variables
metrics
6regularization_losses
layers
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
8:6@2le_net_fcb_n3/conv2d_2/kernel
*:(2le_net_fcb_n3/conv2d_2/bias
.
80
91"
trackable_list_wrapper
.
80
91"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
 layer_regularization_losses
layer_metrics
:trainable_variables
;	variables
non_trainable_variables
metrics
<regularization_losses
layers
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
 layer_regularization_losses
 layer_metrics
>trainable_variables
?	variables
¡non_trainable_variables
¢metrics
@regularization_losses
£layers
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
 ¤layer_regularization_losses
¥layer_metrics
Btrainable_variables
C	variables
¦non_trainable_variables
§metrics
Dregularization_losses
¨layers
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
 ©layer_regularization_losses
ªlayer_metrics
Ftrainable_variables
G	variables
«non_trainable_variables
¬metrics
Hregularization_losses
­layers
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
.:,
ð2le_net_fcb_n3/dense/kernel
':%ð2le_net_fcb_n3/dense/bias
.
J0
K1"
trackable_list_wrapper
.
J0
K1"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
 ®layer_regularization_losses
¯layer_metrics
Ltrainable_variables
M	variables
°non_trainable_variables
±metrics
Nregularization_losses
²layers
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
 ³layer_regularization_losses
´layer_metrics
Ptrainable_variables
Q	variables
µnon_trainable_variables
¶metrics
Rregularization_losses
·layers
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
8:6ð2)le_net_fcb_n3/batch_normalization_1/gamma
7:5ð2(le_net_fcb_n3/batch_normalization_1/beta
@:>ð (2/le_net_fcb_n3/batch_normalization_1/moving_mean
D:Bð (23le_net_fcb_n3/batch_normalization_1/moving_variance
.
U0
V1"
trackable_list_wrapper
<
U0
V1
W2
X3"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
 ¸layer_regularization_losses
¹layer_metrics
Ytrainable_variables
Z	variables
ºnon_trainable_variables
»metrics
[regularization_losses
¼layers
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
7:5T2)le_net_fcb_n3/batch_normalization_2/gamma
6:4T2(le_net_fcb_n3/batch_normalization_2/beta
?:=T (2/le_net_fcb_n3/batch_normalization_2/moving_mean
C:AT (23le_net_fcb_n3/batch_normalization_2/moving_variance
.
^0
_1"
trackable_list_wrapper
<
^0
_1
`2
a3"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
 ½layer_regularization_losses
¾layer_metrics
btrainable_variables
c	variables
¿non_trainable_variables
Àmetrics
dregularization_losses
Álayers
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
/:-	ðT2le_net_fcb_n3/dense_1/kernel
(:&T2le_net_fcb_n3/dense_1/bias
.
f0
g1"
trackable_list_wrapper
.
f0
g1"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
 Âlayer_regularization_losses
Ãlayer_metrics
htrainable_variables
i	variables
Änon_trainable_variables
Åmetrics
jregularization_losses
Ælayers
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
.:,T2le_net_fcb_n3/dense_2/kernel
(:&2le_net_fcb_n3/dense_2/bias
.
l0
m1"
trackable_list_wrapper
.
l0
m1"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
 Çlayer_regularization_losses
Èlayer_metrics
ntrainable_variables
o	variables
Énon_trainable_variables
Êmetrics
pregularization_losses
Ëlayers
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
J
 0
!1
W2
X3
`4
a5"
trackable_list_wrapper
0
Ì0
Í1"
trackable_list_wrapper

0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
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
.
 0
!1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
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
.
W0
X1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
`0
a1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¿

Îtotal

Ïcount
Ð	variables
Ñ	keras_api"
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}


Òtotal

Ócount
Ô
_fn_kwargs
Õ	variables
Ö	keras_api"¸
_tf_keras_metric{"class_name": "MeanMetricWrapper", "name": "accuracy", "dtype": "float32", "config": {"name": "accuracy", "dtype": "float32", "fn": "categorical_accuracy"}}
:  (2total
:  (2count
0
Î0
Ï1"
trackable_list_wrapper
.
Ð	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
Ò0
Ó1"
trackable_list_wrapper
.
Õ	variables"
_generic_user_object
::8 2"Adam/le_net_fcb_n3/conv2d/kernel/m
,:* 2 Adam/le_net_fcb_n3/conv2d/bias/m
::8 2.Adam/le_net_fcb_n3/batch_normalization/gamma/m
9:7 2-Adam/le_net_fcb_n3/batch_normalization/beta/m
<:: @2$Adam/le_net_fcb_n3/conv2d_1/kernel/m
.:,@2"Adam/le_net_fcb_n3/conv2d_1/bias/m
=:;@2$Adam/le_net_fcb_n3/conv2d_2/kernel/m
/:-2"Adam/le_net_fcb_n3/conv2d_2/bias/m
3:1
ð2!Adam/le_net_fcb_n3/dense/kernel/m
,:*ð2Adam/le_net_fcb_n3/dense/bias/m
=:;ð20Adam/le_net_fcb_n3/batch_normalization_1/gamma/m
<::ð2/Adam/le_net_fcb_n3/batch_normalization_1/beta/m
<::T20Adam/le_net_fcb_n3/batch_normalization_2/gamma/m
;:9T2/Adam/le_net_fcb_n3/batch_normalization_2/beta/m
4:2	ðT2#Adam/le_net_fcb_n3/dense_1/kernel/m
-:+T2!Adam/le_net_fcb_n3/dense_1/bias/m
3:1T2#Adam/le_net_fcb_n3/dense_2/kernel/m
-:+2!Adam/le_net_fcb_n3/dense_2/bias/m
::8 2"Adam/le_net_fcb_n3/conv2d/kernel/v
,:* 2 Adam/le_net_fcb_n3/conv2d/bias/v
::8 2.Adam/le_net_fcb_n3/batch_normalization/gamma/v
9:7 2-Adam/le_net_fcb_n3/batch_normalization/beta/v
<:: @2$Adam/le_net_fcb_n3/conv2d_1/kernel/v
.:,@2"Adam/le_net_fcb_n3/conv2d_1/bias/v
=:;@2$Adam/le_net_fcb_n3/conv2d_2/kernel/v
/:-2"Adam/le_net_fcb_n3/conv2d_2/bias/v
3:1
ð2!Adam/le_net_fcb_n3/dense/kernel/v
,:*ð2Adam/le_net_fcb_n3/dense/bias/v
=:;ð20Adam/le_net_fcb_n3/batch_normalization_1/gamma/v
<::ð2/Adam/le_net_fcb_n3/batch_normalization_1/beta/v
<::T20Adam/le_net_fcb_n3/batch_normalization_2/gamma/v
;:9T2/Adam/le_net_fcb_n3/batch_normalization_2/beta/v
4:2	ðT2#Adam/le_net_fcb_n3/dense_1/kernel/v
-:+T2!Adam/le_net_fcb_n3/dense_1/bias/v
3:1T2#Adam/le_net_fcb_n3/dense_2/kernel/v
-:+2!Adam/le_net_fcb_n3/dense_2/bias/v
å2â
__inference__wrapped_model_4521¾
²
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *.¢+
)&
input_1ÿÿÿÿÿÿÿÿÿ  
Ý2Ú
G__inference_le_net_fcb_n3_layer_call_and_return_conditional_losses_6115
G__inference_le_net_fcb_n3_layer_call_and_return_conditional_losses_6011
G__inference_le_net_fcb_n3_layer_call_and_return_conditional_losses_5516
G__inference_le_net_fcb_n3_layer_call_and_return_conditional_losses_5588³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ñ2î
,__inference_le_net_fcb_n3_layer_call_fn_6221
,__inference_le_net_fcb_n3_layer_call_fn_6168
,__inference_le_net_fcb_n3_layer_call_fn_5767
,__inference_le_net_fcb_n3_layer_call_fn_5714³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ê2ç
@__inference_conv2d_layer_call_and_return_conditional_losses_6232¢
²
FullArgSpec
args
jself
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
annotationsª *
 
Ï2Ì
%__inference_conv2d_layer_call_fn_6241¢
²
FullArgSpec
args
jself
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
annotationsª *
 
ö2ó
M__inference_batch_normalization_layer_call_and_return_conditional_losses_6279
M__inference_batch_normalization_layer_call_and_return_conditional_losses_6325
M__inference_batch_normalization_layer_call_and_return_conditional_losses_6261
M__inference_batch_normalization_layer_call_and_return_conditional_losses_6343´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2
2__inference_batch_normalization_layer_call_fn_6292
2__inference_batch_normalization_layer_call_fn_6369
2__inference_batch_normalization_layer_call_fn_6305
2__inference_batch_normalization_layer_call_fn_6356´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
¯2¬
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_4631à
²
FullArgSpec
args
jself
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
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
2
,__inference_max_pooling2d_layer_call_fn_4637à
²
FullArgSpec
args
jself
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
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ì2é
B__inference_conv2d_1_layer_call_and_return_conditional_losses_6380¢
²
FullArgSpec
args
jself
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
annotationsª *
 
Ñ2Î
'__inference_conv2d_1_layer_call_fn_6389¢
²
FullArgSpec
args
jself
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
annotationsª *
 
±2®
I__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_4643à
²
FullArgSpec
args
jself
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
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
2
.__inference_max_pooling2d_1_layer_call_fn_4649à
²
FullArgSpec
args
jself
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
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
î2ë
K__inference_spatial_dropout2d_layer_call_and_return_conditional_losses_6417
K__inference_spatial_dropout2d_layer_call_and_return_conditional_losses_6450
K__inference_spatial_dropout2d_layer_call_and_return_conditional_losses_6455
K__inference_spatial_dropout2d_layer_call_and_return_conditional_losses_6412´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2ÿ
0__inference_spatial_dropout2d_layer_call_fn_6422
0__inference_spatial_dropout2d_layer_call_fn_6427
0__inference_spatial_dropout2d_layer_call_fn_6465
0__inference_spatial_dropout2d_layer_call_fn_6460´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ì2é
B__inference_conv2d_2_layer_call_and_return_conditional_losses_6476¢
²
FullArgSpec
args
jself
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
annotationsª *
 
Ñ2Î
'__inference_conv2d_2_layer_call_fn_6485¢
²
FullArgSpec
args
jself
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
annotationsª *
 
ö2ó
M__inference_spatial_dropout2d_1_layer_call_and_return_conditional_losses_6513
M__inference_spatial_dropout2d_1_layer_call_and_return_conditional_losses_6551
M__inference_spatial_dropout2d_1_layer_call_and_return_conditional_losses_6508
M__inference_spatial_dropout2d_1_layer_call_and_return_conditional_losses_6546´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2
2__inference_spatial_dropout2d_1_layer_call_fn_6556
2__inference_spatial_dropout2d_1_layer_call_fn_6518
2__inference_spatial_dropout2d_1_layer_call_fn_6561
2__inference_spatial_dropout2d_1_layer_call_fn_6523´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
±2®
I__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_4791à
²
FullArgSpec
args
jself
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
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
2
.__inference_max_pooling2d_2_layer_call_fn_4797à
²
FullArgSpec
args
jself
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
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ë2è
A__inference_flatten_layer_call_and_return_conditional_losses_6567¢
²
FullArgSpec
args
jself
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
annotationsª *
 
Ð2Í
&__inference_flatten_layer_call_fn_6572¢
²
FullArgSpec
args
jself
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
annotationsª *
 
é2æ
?__inference_dense_layer_call_and_return_conditional_losses_6583¢
²
FullArgSpec
args
jself
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
annotationsª *
 
Î2Ë
$__inference_dense_layer_call_fn_6592¢
²
FullArgSpec
args
jself
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
annotationsª *
 
À2½
A__inference_dropout_layer_call_and_return_conditional_losses_6604
A__inference_dropout_layer_call_and_return_conditional_losses_6609´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2
&__inference_dropout_layer_call_fn_6619
&__inference_dropout_layer_call_fn_6614´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ü2Ù
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_6675
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_6655´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
¦2£
4__inference_batch_normalization_1_layer_call_fn_6701
4__inference_batch_normalization_1_layer_call_fn_6688´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ü2Ù
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_6737
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_6757´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
¦2£
4__inference_batch_normalization_2_layer_call_fn_6783
4__inference_batch_normalization_2_layer_call_fn_6770´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ë2è
A__inference_dense_1_layer_call_and_return_conditional_losses_6794¢
²
FullArgSpec
args
jself
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
annotationsª *
 
Ð2Í
&__inference_dense_1_layer_call_fn_6803¢
²
FullArgSpec
args
jself
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
annotationsª *
 
ë2è
A__inference_dense_2_layer_call_and_return_conditional_losses_6814¢
²
FullArgSpec
args
jself
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
annotationsª *
 
Ð2Í
&__inference_dense_2_layer_call_fn_6823¢
²
FullArgSpec
args
jself
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
annotationsª *
 
1B/
"__inference_signature_wrapper_5830input_1­
__inference__wrapped_model_4521 !*+89JKWXVUfg`a_^lm8¢5
.¢+
)&
input_1ÿÿÿÿÿÿÿÿÿ  
ª "3ª0
.
output_1"
output_1ÿÿÿÿÿÿÿÿÿ·
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_6655dWXVU4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿð
p
ª "&¢#

0ÿÿÿÿÿÿÿÿÿð
 ·
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_6675dWXVU4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿð
p 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿð
 
4__inference_batch_normalization_1_layer_call_fn_6688WWXVU4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿð
p
ª "ÿÿÿÿÿÿÿÿÿð
4__inference_batch_normalization_1_layer_call_fn_6701WWXVU4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿð
p 
ª "ÿÿÿÿÿÿÿÿÿðµ
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_6737b`a_^3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿT
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿT
 µ
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_6757b`a_^3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿT
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿT
 
4__inference_batch_normalization_2_layer_call_fn_6770U`a_^3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿT
p
ª "ÿÿÿÿÿÿÿÿÿT
4__inference_batch_normalization_2_layer_call_fn_6783U`a_^3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿT
p 
ª "ÿÿÿÿÿÿÿÿÿTÃ
M__inference_batch_normalization_layer_call_and_return_conditional_losses_6261r !;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ 
p
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ 
 Ã
M__inference_batch_normalization_layer_call_and_return_conditional_losses_6279r !;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ 
p 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ 
 è
M__inference_batch_normalization_layer_call_and_return_conditional_losses_6325 !M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 è
M__inference_batch_normalization_layer_call_and_return_conditional_losses_6343 !M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p 
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
2__inference_batch_normalization_layer_call_fn_6292e !;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ 
p
ª " ÿÿÿÿÿÿÿÿÿ 
2__inference_batch_normalization_layer_call_fn_6305e !;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ 
p 
ª " ÿÿÿÿÿÿÿÿÿ À
2__inference_batch_normalization_layer_call_fn_6356 !M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ À
2__inference_batch_normalization_layer_call_fn_6369 !M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p 
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ ²
B__inference_conv2d_1_layer_call_and_return_conditional_losses_6380l*+7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@
 
'__inference_conv2d_1_layer_call_fn_6389_*+7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ 
ª " ÿÿÿÿÿÿÿÿÿ@³
B__inference_conv2d_2_layer_call_and_return_conditional_losses_6476m897¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ
 
'__inference_conv2d_2_layer_call_fn_6485`897¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@
ª "!ÿÿÿÿÿÿÿÿÿ°
@__inference_conv2d_layer_call_and_return_conditional_losses_6232l7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ  
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ 
 
%__inference_conv2d_layer_call_fn_6241_7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ  
ª " ÿÿÿÿÿÿÿÿÿ ¢
A__inference_dense_1_layer_call_and_return_conditional_losses_6794]fg0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿð
ª "%¢"

0ÿÿÿÿÿÿÿÿÿT
 z
&__inference_dense_1_layer_call_fn_6803Pfg0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿð
ª "ÿÿÿÿÿÿÿÿÿT¡
A__inference_dense_2_layer_call_and_return_conditional_losses_6814\lm/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿT
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 y
&__inference_dense_2_layer_call_fn_6823Olm/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿT
ª "ÿÿÿÿÿÿÿÿÿ¡
?__inference_dense_layer_call_and_return_conditional_losses_6583^JK0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿð
 y
$__inference_dense_layer_call_fn_6592QJK0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿð£
A__inference_dropout_layer_call_and_return_conditional_losses_6604^4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿð
p
ª "&¢#

0ÿÿÿÿÿÿÿÿÿð
 £
A__inference_dropout_layer_call_and_return_conditional_losses_6609^4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿð
p 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿð
 {
&__inference_dropout_layer_call_fn_6614Q4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿð
p
ª "ÿÿÿÿÿÿÿÿÿð{
&__inference_dropout_layer_call_fn_6619Q4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿð
p 
ª "ÿÿÿÿÿÿÿÿÿð§
A__inference_flatten_layer_call_and_return_conditional_losses_6567b8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 
&__inference_flatten_layer_call_fn_6572U8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿÊ
G__inference_le_net_fcb_n3_layer_call_and_return_conditional_losses_5516 !*+89JKWXVUfg`a_^lm<¢9
2¢/
)&
input_1ÿÿÿÿÿÿÿÿÿ  
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Ê
G__inference_le_net_fcb_n3_layer_call_and_return_conditional_losses_5588 !*+89JKWXVUfg`a_^lm<¢9
2¢/
)&
input_1ÿÿÿÿÿÿÿÿÿ  
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 É
G__inference_le_net_fcb_n3_layer_call_and_return_conditional_losses_6011~ !*+89JKWXVUfg`a_^lm;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ  
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 É
G__inference_le_net_fcb_n3_layer_call_and_return_conditional_losses_6115~ !*+89JKWXVUfg`a_^lm;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ  
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ¢
,__inference_le_net_fcb_n3_layer_call_fn_5714r !*+89JKWXVUfg`a_^lm<¢9
2¢/
)&
input_1ÿÿÿÿÿÿÿÿÿ  
p
ª "ÿÿÿÿÿÿÿÿÿ¢
,__inference_le_net_fcb_n3_layer_call_fn_5767r !*+89JKWXVUfg`a_^lm<¢9
2¢/
)&
input_1ÿÿÿÿÿÿÿÿÿ  
p 
ª "ÿÿÿÿÿÿÿÿÿ¡
,__inference_le_net_fcb_n3_layer_call_fn_6168q !*+89JKWXVUfg`a_^lm;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ  
p
ª "ÿÿÿÿÿÿÿÿÿ¡
,__inference_le_net_fcb_n3_layer_call_fn_6221q !*+89JKWXVUfg`a_^lm;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ  
p 
ª "ÿÿÿÿÿÿÿÿÿì
I__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_4643R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Ä
.__inference_max_pooling2d_1_layer_call_fn_4649R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿì
I__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_4791R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Ä
.__inference_max_pooling2d_2_layer_call_fn_4797R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿê
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_4631R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Â
,__inference_max_pooling2d_layer_call_fn_4637R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ»
"__inference_signature_wrapper_5830 !*+89JKWXVUfg`a_^lmC¢@
¢ 
9ª6
4
input_1)&
input_1ÿÿÿÿÿÿÿÿÿ  "3ª0
.
output_1"
output_1ÿÿÿÿÿÿÿÿÿô
M__inference_spatial_dropout2d_1_layer_call_and_return_conditional_losses_6508¢V¢S
L¢I
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ô
M__inference_spatial_dropout2d_1_layer_call_and_return_conditional_losses_6513¢V¢S
L¢I
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ¿
M__inference_spatial_dropout2d_1_layer_call_and_return_conditional_losses_6546n<¢9
2¢/
)&
inputsÿÿÿÿÿÿÿÿÿ
p
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ
 ¿
M__inference_spatial_dropout2d_1_layer_call_and_return_conditional_losses_6551n<¢9
2¢/
)&
inputsÿÿÿÿÿÿÿÿÿ
p 
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ
 Ì
2__inference_spatial_dropout2d_1_layer_call_fn_6518V¢S
L¢I
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÌ
2__inference_spatial_dropout2d_1_layer_call_fn_6523V¢S
L¢I
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
2__inference_spatial_dropout2d_1_layer_call_fn_6556a<¢9
2¢/
)&
inputsÿÿÿÿÿÿÿÿÿ
p
ª "!ÿÿÿÿÿÿÿÿÿ
2__inference_spatial_dropout2d_1_layer_call_fn_6561a<¢9
2¢/
)&
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "!ÿÿÿÿÿÿÿÿÿ»
K__inference_spatial_dropout2d_layer_call_and_return_conditional_losses_6412l;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ@
p
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@
 »
K__inference_spatial_dropout2d_layer_call_and_return_conditional_losses_6417l;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ@
p 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@
 ò
K__inference_spatial_dropout2d_layer_call_and_return_conditional_losses_6450¢V¢S
L¢I
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ò
K__inference_spatial_dropout2d_layer_call_and_return_conditional_losses_6455¢V¢S
L¢I
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
0__inference_spatial_dropout2d_layer_call_fn_6422_;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ@
p
ª " ÿÿÿÿÿÿÿÿÿ@
0__inference_spatial_dropout2d_layer_call_fn_6427_;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ@
p 
ª " ÿÿÿÿÿÿÿÿÿ@Ê
0__inference_spatial_dropout2d_layer_call_fn_6460V¢S
L¢I
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÊ
0__inference_spatial_dropout2d_layer_call_fn_6465V¢S
L¢I
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ