function out = model(alpha, k, defo)

% defo - limitation of drforamions in ones[1]
% k - mesh size
% alphs - The number of the geomety in [mm]

defo = 0.1
k = 5
alpha = 3618

import com.comsol.model.*
import com.comsol.model.util.*

model = ModelUtil.create('model');
model.modelPath('E:...\'); % CHANGE

model.component.create('comp1', true);

%----------------------------------------------------
%-----------------------parameters-------------------
%----------------------------------------------------

model.param.set('a', '0.01[m]');
model.param.set('pois', '0.3');
model.param.set('YouM', '110e9[Pa]');
model.param.set('dens', '4430[kg/(m^3)]');

model.param.set('yiel', '871[MPa]');
model.param.set('fai', '1057[MPa]');
model.param.set('def', defo);
model.param.set('strain', '0.05');

%----------------------------------------------------
%-----------------------geometry---------------------
%----------------------------------------------------
model.component('comp1').geom.create('geom1', 3);
model.component('comp1').geom('geom1').lengthUnit('m');

model.component('comp1').geom('geom1').selection.create('csel1', 'CumulativeSelection');
model.component('comp1').geom('geom1').selection('csel1').label('dom');

model.component('comp1').geom('geom1').create('imp1', 'Import');
model.component('comp1').geom('geom1').feature('imp1').set('filename', "E:...\"+alpha+".stp");
model.component('comp1').geom('geom1').feature('imp1').set('contributeto', 'csel1');
model.component('comp1').geom('geom1').run;

%-------------------------stl-----------------------
model.component('comp1').geom('geom1').create('boxsel1', 'BoxSelection');
model.component('comp1').geom('geom1').feature('boxsel1').set('condition', 'allvertices');
model.component('comp1').geom('geom1').feature('boxsel1').set('entitydim', 2);
model.component('comp1').geom('geom1').feature('boxsel1').set('zmin', '-a');
model.component('comp1').geom('geom1').feature('boxsel1').set('zmax', '-a/2+0.000001');
%model.component('comp1').geom('geom1').feature('boxsel1').set('ymin', '-a');
%model.component('comp1').geom('geom1').feature('boxsel1').set('ymax', '-a/2');

model.component('comp1').geom('geom1').create('boxsel2', 'BoxSelection');
model.component('comp1').geom('geom1').feature('boxsel2').set('condition', 'allvertices');
model.component('comp1').geom('geom1').feature('boxsel2').set('entitydim', 2);
model.component('comp1').geom('geom1').feature('boxsel2').set('zmin', 'a/2-0.0000001');
model.component('comp1').geom('geom1').feature('boxsel2').set('zmax', 'a');
%model.component('comp1').geom('geom1').feature('boxsel2').set('ymin', 'a/2');
%model.component('comp1').geom('geom1').feature('boxsel2').set('ymax', 'a');

model.component('comp1').geom('geom1').run;


%----------------------------------------------------
%-------------material-properties--------------------
%----------------------------------------------------

model.component('comp1').material.create('mat1', 'Common');
model.component('comp1').material('mat1').propertyGroup.create('Enu', 'Young''s_modulus_and_Poisson''s_ratio');
model.component('comp1').material('mat1').propertyGroup('Enu').set('E', {'YouM'});
model.component('comp1').material('mat1').propertyGroup('Enu').set('nu', {'pois'});
model.component('comp1').material('mat1').propertyGroup('def').set('density', {'dens'});
model.component('comp1').material('mat1').selection.named('geom1_csel1_dom');


%----------------------------------------------------
%--------------solid-mechanics-press-----------------
%----------------------------------------------------
model.component('comp1').physics.create('solid', 'SolidMechanics', 'geom1');
model.component('comp1').physics('solid').feature('lemm1').create('plsty1', 'Plasticity', 3);
model.component('comp1').physics('solid').feature('lemm1').feature('plsty1').set('IsotropicHardeningModel', 'HardeningFunction');


model.component('comp1').physics('solid').create('disp1', 'Displacement2', 2);
model.component('comp1').physics('solid').feature('disp1').selection.named('geom1_boxsel2');
model.component('comp1').physics('solid').feature('disp1').setIndex('Direction', 'prescribed', 1);
model.component('comp1').physics('solid').feature('disp1').setIndex('U0', '-a*strain', 1);
model.component('comp1').physics('solid').create('fix1', 'Fixed', 2);
model.component('comp1').physics('solid').feature('fix1').selection.named('geom1_boxsel1');


%mphsave(model,'D:\s0\test_astr');


%-------yield-stressstrain-stress-elasticity---------
model.component('comp1').material('mat1').propertyGroup.create('ElastoplasticModel', 'Elastoplastic_material_model');
model.component('comp1').material('mat1').propertyGroup('ElastoplasticModel').set('sigmags', {'yiel'});
model.component('comp1').material('mat1').propertyGroup('ElastoplasticModel').func.create('int1', 'Interpolation');
model.component('comp1').material('mat1').propertyGroup('ElastoplasticModel').func('int1').set('table', {'0.0093' '871';  ...
'0.00981' '891';  ...
'0.01' '898';  ...
'0.00105' '916';  ...
'0.013' '978';  ...
'0.01402' '996';  ...
'0.015' '1011';  ...
'0.0173' '1036';  ...
'0.022' '1070';  ...
'0.03' '1094'; ...
'0.05' '1087';  ...
'0.07' '1019';  ...
'0.087' '920';  ...
'0.088' '894'});
model.component('comp1').material('mat1').propertyGroup('ElastoplasticModel').func('int1').setIndex('fununit', '[MPa]', 0);
model.component('comp1').material('mat1').propertyGroup('ElastoplasticModel').func('int1').setIndex('argunit', 1, 0);
model.component('comp1').material('mat1').propertyGroup('ElastoplasticModel').func('int1').setIndex('fununit', 'MPa', 0);
model.component('comp1').material('mat1').propertyGroup('ElastoplasticModel').addInput('effectiveplasticstrain');
model.component('comp1').material('mat1').propertyGroup('ElastoplasticModel').set('sigmagh', {'int1(epe)'});


%------von-Mises-failure-model--------
model.component('comp1').physics('solid').feature('lemm1').create('sf1', 'Safety', 3);

model.component('comp1').material('mat1').propertyGroup.create('IsotropicStrengthParameters', 'Isotropic_strength_parameters');
model.component('comp1').material('mat1').propertyGroup('IsotropicStrengthParameters').set('sigmat', {'fai'});
%----------------------------------------------------
%-------------------------mesh-----------------------
%----------------------------------------------------

model.component('comp1').mesh.create('mesh1');
model.component('comp1').mesh('mesh1').autoMeshSize(k);

%----------------------------------------------------
%-----------------study-stationary-------------------
%----------------------------------------------------


model.study.create('std1');
model.study('std1').create('stat', 'Stationary');
model.study('std1').feature('stat').setSolveFor('/physics/solid', true);
model.study('std1').create('param', 'Parametric');
model.study('std1').feature('param').setIndex('pname', 'a', 0);
model.study('std1').feature('param').setIndex('plistarr', '', 0);
model.study('std1').feature('param').setIndex('punit', 'm', 0);
model.study('std1').feature('param').setIndex('pname', 'a', 0);
model.study('std1').feature('param').setIndex('plistarr', '', 0);
model.study('std1').feature('param').setIndex('punit', 'm', 0);
model.study('std1').feature('param').setIndex('pname', 'strain', 0);
model.study('std1').feature('param').setIndex('plistarr', 'range(0,1e-3,def)', 0);


%mphsave(model,'D:\s0\test_astr');
%----------------------------------------------------
%---------------------processing---------------------
%----------------------------------------------------

model.sol.create('sol1');
model.sol('sol1').study('std1');
model.sol('sol1').create('st1', 'StudyStep');
model.sol('sol1').feature('st1').set('study', 'std1');
model.sol('sol1').feature('st1').set('studystep', 'stat');
model.sol('sol1').create('v1', 'Variables');
model.sol('sol1').feature('v1').set('control', 'stat');
model.sol('sol1').create('s1', 'Stationary');
model.sol('sol1').feature('s1').create('p1', 'Parametric');
model.sol('sol1').feature('s1').feature('p1').set('pname', {'strain'});
model.sol('sol1').feature('s1').feature('p1').set('plistarr', {'range(0,1.0e-4,def)'});
model.sol('sol1').feature('s1').feature('p1').set('punit', {''});
model.sol('sol1').feature('s1').feature('p1').set('sweeptype', 'sparse');
model.sol('sol1').feature('s1').feature('p1').set('preusesol', 'no');
model.sol('sol1').feature('s1').feature('p1').set('pcontinuationmode', 'no');
model.sol('sol1').feature('s1').feature('p1').set('porder', 'constant');
model.sol('sol1').feature('s1').feature('p1').set('plot', 'off');
model.sol('sol1').feature('s1').feature('p1').set('plotgroup', 'Default');
model.sol('sol1').feature('s1').feature('p1').set('probesel', 'all');
model.sol('sol1').feature('s1').feature('p1').set('probes', {});
model.sol('sol1').feature('s1').feature('p1').set('control', 'param');
model.sol('sol1').feature('s1').set('control', 'stat');
model.sol('sol1').feature('s1').feature('aDef').set('cachepattern', true);
model.sol('sol1').feature('s1').create('fc1', 'FullyCoupled');
model.sol('sol1').feature('s1').feature('fc1').set('termonres', 'auto');
model.sol('sol1').feature('s1').feature('fc1').set('reserrfact', 1000);
model.sol('sol1').feature('s1').create('d1', 'Direct');
model.sol('sol1').feature('s1').feature('d1').set('linsolver', 'pardiso');
model.sol('sol1').feature('s1').feature('d1').set('pivotperturb', 1.0E-9);
model.sol('sol1').feature('s1').feature('d1').set('nliniterrefine', true);
model.sol('sol1').feature('s1').feature('d1').label('Suggested Direct Solver (solid)');
model.sol('sol1').feature('s1').create('i1', 'Iterative');
model.sol('sol1').feature('s1').feature('i1').set('linsolver', 'gmres');
model.sol('sol1').feature('s1').feature('i1').set('rhob', 40);
model.sol('sol1').feature('s1').feature('i1').set('nlinnormuse', true);
model.sol('sol1').feature('s1').feature('i1').label('Suggested Iterative Solver (solid)');
model.sol('sol1').feature('s1').feature('i1').create('mg1', 'Multigrid');
model.sol('sol1').feature('s1').feature('i1').feature('mg1').set('prefun', 'gmg');
model.sol('sol1').feature('s1').feature('i1').feature('mg1').feature('pr').create('so1', 'SOR');
model.sol('sol1').feature('s1').feature('i1').feature('mg1').feature('pr').feature('so1').set('relax', 0.8);
model.sol('sol1').feature('s1').feature('i1').feature('mg1').feature('po').create('so1', 'SOR');
model.sol('sol1').feature('s1').feature('i1').feature('mg1').feature('po').feature('so1').set('relax', 0.8);
model.sol('sol1').feature('s1').feature('i1').feature('mg1').feature('cs').create('d1', 'Direct');
model.sol('sol1').feature('s1').feature('i1').feature('mg1').feature('cs').feature('d1').set('linsolver', 'pardiso');
model.sol('sol1').feature('s1').feature('i1').feature('mg1').feature('cs').feature('d1').set('pivotperturb', 1.0E-9);
model.sol('sol1').feature('s1').feature('fc1').set('linsolver', 'd1');
model.sol('sol1').feature('s1').feature('fc1').set('termonres', 'auto');
model.sol('sol1').feature('s1').feature('fc1').set('reserrfact', 1000);
model.sol('sol1').feature('s1').feature.remove('fcDef');
model.sol('sol1').attach('std1');
model.sol('sol1').runAll;


%----------------------------------------------------
%-----------------------results----------------------
%----------------------------------------------------


model.result.numerical.create('av1', 'AvVolume');
model.result.numerical('av1').selection.set([1]);
model.result.numerical('av1').setIndex('looplevelinput', 'all', 0);
model.result.numerical('av1').set('expr', {'solid.misesGp' '-solid.RFtotalz' 'solid.lemm1.sf1.d_i' 'solid.lemm1.sf1.f_i'});

model.result.table.create('tbl1', 'Table');
model.result.table('tbl1').comments('Volume Average 1');
model.result.numerical('av1').set('table', 'tbl1');
model.result.numerical('av1').setResult;

model.result.numerical.create('av2', 'AvVolume');
model.result.numerical('av2').setIndex('looplevelinput', 'first', 0);
model.result.numerical('av2').selection.set([1]);
model.result.numerical('av2').set('expr', {'solid.sYGp'});

model.result.table.create('tbl2', 'Table');
model.result.table('tbl2').comments('Volume Average 2');
model.result.numerical('av2').set('table', 'tbl2');
model.result.numerical('av2').setResult;

model.result.numerical.create('meas1', 'MeasureSurface');
%model.result.numerical('meas2').set('intvolume', true);
model.result.numerical('meas1').setIndex('looplevelinput', 'first', 0);
model.result.numerical('meas1').selection.named('geom1_boxsel2');
model.result.numerical('meas1').set('table', 'tbl2');
model.result.numerical('meas1').appendResult;

model.result.numerical.create('meas2', 'MeasureVolume');
model.result.numerical('meas2').selection.set([1]);
model.result.numerical('meas2').set('table', 'tbl2');
model.result.numerical('meas2').appendResult;
%-------------------------txt-----------------------
model.result.export.create('tbl1', 'tbl1', 'Table');
model.result.export('tbl1').set('filename', "E:...\"+alpha+"_ssc.txt").run;

model.result.export.create('tbl2', 'tbl2', 'Table');
model.result.export('tbl2').set('filename', "E:...\"+alpha+"_yv.txt");
model.result.export('tbl2').run;

%mphsave(model,'D:\s0\test_astr');


out = model;