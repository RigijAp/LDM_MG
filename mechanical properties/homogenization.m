function out = model(alpha, k)
% k - mesh size
% alpha - number of the structure in the database/dataset
% a - dimension of elementary lattice cell in [mm]

a=10


import com.comsol.model.*
import com.comsol.model.util.*

model = ModelUtil.create('model');

%%%%%%%%%%%%%%%%%%%%%%%%%%-CHANGE!!!!!!!-%%%%%%%%%%%%%%%%%%%%%%%%%%
model.modelPath('E:\....'); % CHANGE

model.component.create('comp1', true);

%----------------------------------------------------
%-----------------------parameters-------------------
%----------------------------------------------------

model.param.set('a', a);
model.param.set('pois', '0.3');
model.param.set('YouM', '110e9[Pa]');
model.param.set('dens', '4430[kg/(m^3)]');

%----------------------------------------------------
%-----------------------geometry---------------------
%----------------------------------------------------
model.component('comp1').geom.create('geom1', 3);
model.component('comp1').geom('geom1').lengthUnit('mm');

model.component('comp1').geom('geom1').selection.create('csel1', 'CumulativeSelection');
model.component('comp1').geom('geom1').selection('csel1').label('x_pl');
model.component('comp1').geom('geom1').selection.create('csel2', 'CumulativeSelection');
model.component('comp1').geom('geom1').selection('csel2').label('y_pl');
model.component('comp1').geom('geom1').selection.create('csel3', 'CumulativeSelection');
model.component('comp1').geom('geom1').selection('csel3').label('z_pl');
model.component('comp1').geom('geom1').selection.create('csel4', 'CumulativeSelection');
model.component('comp1').geom('geom1').selection('csel4').label('mat');
model.component('comp1').geom('geom1').selection.create('csel5', 'CumulativeSelection');
model.component('comp1').geom('geom1').selection('csel5').label('air');

model.component('comp1').geom('geom1').create('imp1', 'Import');
%%%%%%%%%%%%%%%%%%%%%%%%%%-CHANGE!!!!!!!-%%%%%%%%%%%%%%%%%%%%%%%%%%
model.component('comp1').geom('geom1').feature('imp1').set('filename', "'E:\....'\"+alpha+".stp");
model.component('comp1').geom('geom1').feature('imp1').set('selresult', true);
model.component('comp1').geom('geom1').feature('imp1').set('contributeto', 'csel4');
model.component('comp1').geom('geom1').run;

model.component('comp1').geom('geom1').create('blk1', 'Block');
model.component('comp1').geom('geom1').feature('blk1').set('size', [a a a]);
%model.component('comp1').geom('geom1').feature('blk1').set('base', 'corner');
model.component('comp1').geom('geom1').feature('blk1').set('base', 'center');
model.component('comp1').geom('geom1').feature('blk1').set('pos', [0 0 0]);

model.component('comp1').geom('geom1').create('dif1', 'Difference');
model.component('comp1').geom('geom1').feature('dif1').selection('input').set({'blk1'});
model.component('comp1').geom('geom1').feature('dif1').selection('input2').set({'imp1'});
model.component('comp1').geom('geom1').feature('dif1').set('intbnd', false);
model.component('comp1').geom('geom1').feature('dif1').set('keepsubtract', true);
model.component('comp1').geom('geom1').feature('dif1').set('selresult', true);
model.component('comp1').geom('geom1').feature('dif1').set('contributeto', 'csel5');



model.component('comp1').geom('geom1').create('boxsel1', 'BoxSelection');
model.component('comp1').geom('geom1').feature('boxsel1').set('condition', 'allvertices');
model.component('comp1').geom('geom1').feature('boxsel1').set('entitydim', 2);
%model.component('comp1').geom('geom1').feature('boxsel1').set('xmin', '-2*a');
%model.component('comp1').geom('geom1').feature('boxsel1').set('xmax', '0');
model.component('comp1').geom('geom1').feature('boxsel1').set('xmin', '-a');
model.component('comp1').geom('geom1').feature('boxsel1').set('xmax', '-a/2');
model.component('comp1').geom('geom1').feature('boxsel1').set('contributeto', 'csel1');

model.component('comp1').geom('geom1').create('boxsel2', 'BoxSelection');
model.component('comp1').geom('geom1').feature('boxsel2').set('condition', 'allvertices');
model.component('comp1').geom('geom1').feature('boxsel2').set('entitydim', 2);
%model.component('comp1').geom('geom1').feature('boxsel2').set('xmin', 'a');
%model.component('comp1').geom('geom1').feature('boxsel2').set('xmax', '2*a');
model.component('comp1').geom('geom1').feature('boxsel2').set('xmin', 'a/2');
model.component('comp1').geom('geom1').feature('boxsel2').set('xmax', 'a');
model.component('comp1').geom('geom1').feature('boxsel2').set('contributeto', 'csel1');

model.component('comp1').geom('geom1').create('boxsel3', 'BoxSelection');
model.component('comp1').geom('geom1').feature('boxsel3').set('condition', 'allvertices');
model.component('comp1').geom('geom1').feature('boxsel3').set('entitydim', 2);
%model.component('comp1').geom('geom1').feature('boxsel3').set('ymin', '-a');
%model.component('comp1').geom('geom1').feature('boxsel3').set('ymax', '0');
model.component('comp1').geom('geom1').feature('boxsel3').set('ymin', '-a');
model.component('comp1').geom('geom1').feature('boxsel3').set('ymax', '-a/2');
model.component('comp1').geom('geom1').feature('boxsel3').set('contributeto', 'csel2');

model.component('comp1').geom('geom1').create('boxsel4', 'BoxSelection');
model.component('comp1').geom('geom1').feature('boxsel4').set('condition', 'allvertices');
model.component('comp1').geom('geom1').feature('boxsel4').set('entitydim', 2);
%model.component('comp1').geom('geom1').feature('boxsel4').set('ymax', '2*a');
%model.component('comp1').geom('geom1').feature('boxsel4').set('ymin', 'a');
model.component('comp1').geom('geom1').feature('boxsel4').set('ymax', 'a');
model.component('comp1').geom('geom1').feature('boxsel4').set('ymin', 'a/2');
model.component('comp1').geom('geom1').feature('boxsel4').set('contributeto', 'csel2');

model.component('comp1').geom('geom1').create('boxsel5', 'BoxSelection');
model.component('comp1').geom('geom1').feature('boxsel5').set('condition', 'allvertices');
model.component('comp1').geom('geom1').feature('boxsel5').set('entitydim', 2);
%model.component('comp1').geom('geom1').feature('boxsel5').set('zmin', '-a');
%model.component('comp1').geom('geom1').feature('boxsel5').set('zmax', '0');
model.component('comp1').geom('geom1').feature('boxsel5').set('zmin', '-a');
model.component('comp1').geom('geom1').feature('boxsel5').set('zmax', '-a/2');
model.component('comp1').geom('geom1').feature('boxsel5').set('contributeto', 'csel3');

model.component('comp1').geom('geom1').create('boxsel6', 'BoxSelection');
model.component('comp1').geom('geom1').feature('boxsel6').set('condition', 'allvertices');
model.component('comp1').geom('geom1').feature('boxsel6').set('entitydim', 2);
%model.component('comp1').geom('geom1').feature('boxsel6').set('zmin', 'a');
%model.component('comp1').geom('geom1').feature('boxsel6').set('zmax', 'a*2');
model.component('comp1').geom('geom1').feature('boxsel6').set('zmin', 'a/2');
model.component('comp1').geom('geom1').feature('boxsel6').set('zmax', 'a');
model.component('comp1').geom('geom1').feature('boxsel6').set('contributeto', 'csel3');

model.component('comp1').geom('geom1').run;


%----------------------------------------------------
%-------------material-properties--------------------
%----------------------------------------------------

model.component('comp1').material.create('mat1', 'Common');
model.component('comp1').material('mat1').propertyGroup.create('Enu', 'Young''s_modulus_and_Poisson''s_ratio');
model.component('comp1').material('mat1').propertyGroup('Enu').set('E', {'YouM'});
model.component('comp1').material('mat1').propertyGroup('Enu').set('nu', {'pois'});
model.component('comp1').material('mat1').propertyGroup('def').set('density', {'dens'});
model.component('comp1').material('mat1').selection.named('geom1_csel4_dom');


model.component('comp1').material.create('mat2', 'Common');
model.component('comp1').material('mat2').propertyGroup.create('Enu', 'Young''s_modulus_and_Poisson''s_ratio');
model.component('comp1').material('mat2').propertyGroup('Enu').set('E', {'1'});
model.component('comp1').material('mat2').propertyGroup('Enu').set('nu', {'1.3'});
model.component('comp1').material('mat2').propertyGroup('def').set('density', {'1.5'});
model.component('comp1').material('mat2').selection.named('geom1_csel5_dom');

%----------------------------------------------------
%--------solid-mechanics-(CellPeriodicity)-----------
%----------------------------------------------------

model.component('comp1').physics.create('solid', 'SolidMechanics', 'geom1');
model.component('comp1').physics('solid').create('cp1', 'CellPeriodicity', 3);
model.component('comp1').physics('solid').feature('cp1').set('BoundaryExpansion', 'PrescribedStrain');
model.component('comp1').physics('solid').feature('cp1').set('EffectivePropertiese', 'ElasticityMatrixStandard');
model.component('comp1').physics('solid').feature('cp1').set('Volume', 'Porosity');
model.component('comp1').physics('solid').feature('cp1').set('f', 0.5);
model.component('comp1').physics('solid').feature('cp1').runCommand('createLoadGroupsandStudy');
model.component('comp1').physics('solid').feature('cp1').create('bp1', 'BoundaryPair', 2);
model.component('comp1').physics('solid').feature('cp1').feature('bp1').selection.named('geom1_csel1_bnd');
model.component('comp1').physics('solid').feature('cp1').create('bp2', 'BoundaryPair', 2);
model.component('comp1').physics('solid').feature('cp1').feature('bp2').selection.named('geom1_csel2_bnd');
model.component('comp1').physics('solid').feature('cp1').create('bp3', 'BoundaryPair', 2);
model.component('comp1').physics('solid').feature('cp1').feature('bp3').selection.named('geom1_csel3_bnd');

%----------------------------------------------------
%-------------------------mesh-----------------------
%----------------------------------------------------

model.component('comp1').mesh.create('mesh1');
model.component('comp1').mesh('mesh1').autoMeshSize(k);

%mphsave(model,"'E:\....'\test);

mphsave(model,"'E:\....\homogen' + alpha);


%----------------------------------------------------
%---------------------processing---------------------
%----------------------------------------------------
model.sol('solidcp1sol').study('solidcp1std');
model.sol('solidcp1sol').feature.remove('s1');
model.sol('solidcp1sol').feature.remove('v1');
model.sol('solidcp1sol').feature.remove('st1');
model.sol('solidcp1sol').create('st1', 'StudyStep');
model.sol('solidcp1sol').feature('st1').set('study', 'solidcp1std');
model.sol('solidcp1sol').feature('st1').set('studystep', 'solidcp1stat');
model.sol('solidcp1sol').create('v1', 'Variables');
model.sol('solidcp1sol').feature('v1').set('control', 'solidcp1stat');
model.sol('solidcp1sol').create('s1', 'Stationary');
model.sol('solidcp1sol').feature('s1').create('p1', 'Parametric');
model.sol('solidcp1sol').feature('s1').feature.remove('pDef');
model.sol('solidcp1sol').feature('s1').feature('p1').set('control', 'solidcp1stat');
model.sol('solidcp1sol').feature('s1').set('control', 'solidcp1stat');
model.sol('solidcp1sol').feature('s1').feature('aDef').set('cachepattern', true);
model.sol('solidcp1sol').feature('s1').create('fc1', 'FullyCoupled');
model.sol('solidcp1sol').feature('s1').feature('fc1').set('termonres', 'auto');
model.sol('solidcp1sol').feature('s1').feature('fc1').set('reserrfact', 1000);
model.sol('solidcp1sol').feature('s1').create('d1', 'Direct');
model.sol('solidcp1sol').feature('s1').feature('d1').set('linsolver', 'mumps');
model.sol('solidcp1sol').feature('s1').feature('d1').label('Suggested Direct Solver (solid)');
model.sol('solidcp1sol').feature('s1').create('i1', 'Iterative');
model.sol('solidcp1sol').feature('s1').feature('i1').set('linsolver', 'fgmres');
model.sol('solidcp1sol').feature('s1').feature('i1').set('rhob', 400);
model.sol('solidcp1sol').feature('s1').feature('i1').set('nlinnormuse', true);
model.sol('solidcp1sol').feature('s1').feature('i1').label('Suggested Iterative Solver (solid)');
model.sol('solidcp1sol').feature('s1').feature('i1').create('mg1', 'Multigrid');
model.sol('solidcp1sol').feature('s1').feature('i1').feature('mg1').set('prefun', 'gmg');
model.sol('solidcp1sol').feature('s1').feature('i1').feature('mg1').feature('pr').create('so1', 'SOR');
model.sol('solidcp1sol').feature('s1').feature('i1').feature('mg1').feature('pr').feature('so1').set('relax', 0.8);
model.sol('solidcp1sol').feature('s1').feature('i1').feature('mg1').feature('po').create('so1', 'SOR');
model.sol('solidcp1sol').feature('s1').feature('i1').feature('mg1').feature('po').feature('so1').set('relax', 0.8);
model.sol('solidcp1sol').feature('s1').feature('i1').feature('mg1').feature('cs').create('d1', 'Direct');
model.sol('solidcp1sol').feature('s1').feature('i1').feature('mg1').feature('cs').feature('d1').set('linsolver', 'pardiso');
model.sol('solidcp1sol').feature('s1').feature('i1').feature('mg1').feature('cs').feature('d1').set('pivotperturb', 1.0E-9);
model.sol('solidcp1sol').feature('s1').feature('fc1').set('linsolver', 'd1');
model.sol('solidcp1sol').feature('s1').feature('fc1').set('termonres', 'auto');
model.sol('solidcp1sol').feature('s1').feature('fc1').set('reserrfact', 1000);
model.sol('solidcp1sol').feature('s1').feature.remove('fcDef');
model.sol('solidcp1sol').attach('solidcp1std');
model.sol('solidcp1sol').runAll;

%----------------------------------------------------
%-----------------------results----------------------
%----------------------------------------------------

model.result.numerical.create('av1', 'AvVolume');
model.result.numerical('av1').set('data', 'dset1');
model.result.numerical('av1').setIndex('looplevelinput', 'last', 0);
%model.result.numerical('av1').selection.set([1 2]);
model.result.numerical('av1').selection.all;
model.result.numerical('av1').set('expr', {'solid.rho' '-1/solid.cp1.Dinv11*solid.cp1.Dinv12' '1/solid.cp1.Dinv11' '1/solid.cp1.Dinv55' 'solid.K' 'solid.trescaGp'  ...
    'solid.cp1.D11' 'solid.cp1.D12' 'solid.cp1.D13' 'solid.cp1.D14' 'solid.cp1.D15' 'solid.cp1.D16' 'solid.cp1.D12' 'solid.cp1.D22' 'solid.cp1.D23' 'solid.cp1.D24' 'solid.cp1.D25' 'solid.cp1.D26' ...
    'solid.cp1.D13' 'solid.cp1.D23' 'solid.cp1.D33' 'solid.cp1.D34' 'solid.cp1.D35' 'solid.cp1.D36' 'solid.cp1.D14' 'solid.cp1.D24' 'solid.cp1.D34' 'solid.cp1.D44' 'solid.cp1.D45' 'solid.cp1.D46' ...
    'solid.cp1.D15' 'solid.cp1.D25' 'solid.cp1.D35' 'solid.cp1.D45' 'solid.cp1.D55' 'solid.cp1.D56' 'solid.cp1.D16' 'solid.cp1.D26' 'solid.cp1.D36' 'solid.cp1.D46' 'solid.cp1.D56' 'solid.cp1.D66' ...
    'solid.cp1.Dinv11' 'solid.cp1.Dinv12' 'solid.cp1.Dinv13' 'solid.cp1.Dinv21' 'solid.cp1.Dinv22' 'solid.cp1.Dinv23' 'solid.cp1.Dinv31' 'solid.cp1.Dinv32' 'solid.cp1.Dinv33' 'solid.cp1.Dinv44' 'solid.cp1.Dinv55' 'solid.cp1.Dinv66'});

%----------------------------------------------------
%-------------------save-results---------------------
%----------------------------------------------------

%-------------------------txt-----------------------
model.result.table.create('tbl1', 'Table');
model.result.table('tbl1').comments('Volume Average 1');
model.result.numerical('av1').set('table', 'tbl1');
model.result.numerical('av1').setResult;


model.result.export.create('tbl1', 'Table');
model.result.export('tbl1').set('filename', "'E:\....'\out\"+alpha+"stl_TC4.txt");%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% CHANGE path
model.result.export('tbl1').run;

%mphsave(model,"'E:\....'\test_homo"+alpha);
out = model;