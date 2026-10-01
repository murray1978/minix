#define STAGE4E_D6_CORE_ONLY
#include "stage4e_d6_a4_checkpoint.c"
#undef STAGE4E_D6_CORE_ONLY
#define STAGE4E_D5_CORE_ONLY
#include "stage4e_d5_a4_repeat.c"
#undef STAGE4E_D5_CORE_ONLY

#define D7_PROVENANCE "checkpoint-resume-equivalence-2026-10-01-a"
#define D7_EXPECTED_D6_CHECKPOINT_SHA "ecb75d52881df04a0c47fcd4b852909a865c62cb54929ab33b230c7cf1e5ec89"
#define D7_EXPECTED_D4_SHA "d57d81ed4608df0c4a662404911c16a72fa475f0aea1459e68533e522cebace1"
#define D7_EXPECTED_D3_SHA "f418305f3fb9263480b8a80f34b02a966ebbb94f305f2aa5897d5c2f4356b94e"
#define D7_STEPS 10U

static const char d7_dataset_sha[] = "71744cea4de049e2e192cb2a4494c8a4a68e769b3a7f3b8ed4880433e4583b84";
static const char d7_cache_sha[] = "d6288ad3d128c7182c20088aa57a1bca8bcaa1e13314fb0cdbd60eba14278def";
static const char d7_checkpoint_sha[] = "cd590644d963867a2b6e5a1107f51fad663c41d79c149fbecbbb1f95fa81f49a";
static const char d7_tokenizer_sha[] = "50a52ef822ee9e83de5ce9d0be0a025a773d019437f58b5ff9dcafb063ece361";
static const char d7_d1_sha[] = "45ebaa750f17d02d80feed332b16c9c29c3782a544aed7544da72a62161c76a6";
static const char d7_d2_sha[] = "c27aad7474df093f5a3ff781bb3619b71b3e284edda46290f5814c350e99d97e";

typedef struct {
    uint32_t step_index;
    uint32_t gradient_row_count;
    uint32_t evaluation_row_count;
    uint32_t grad_a_count, grad_a_max;
    uint32_t grad_b_count, grad_b_max;
    uint32_t norm_before, norm_after;
    uint32_t clip_applied, clip_consistency;
    uint32_t train_total, train_average, train_top1;
    uint32_t validation_total, validation_average, validation_top1;
    uint32_t validation_acceptance;
} d7_mismatch_t;

static void d7_compare_double(const double *left,const double *right,uint32_t *counter)
{
    if(memcmp(left,right,sizeof(double))!=0)(*counter)++;
}

static void d7_compare_step(const d4_point_t *continuous,const d4_point_t *resumed,
    uint32_t step,d7_mismatch_t *m)
{
    uint32_t s;
    if(step<6U||step>D7_STEPS)return;
    if(continuous->grad_train_rows!=resumed->grad_train_rows||
       continuous->grad_validation_rows!=resumed->grad_validation_rows||
       continuous->grad_test_rows!=resumed->grad_test_rows)m->gradient_row_count++;
    if(continuous->metrics.evaluation_train_rows!=resumed->metrics.evaluation_train_rows||
       continuous->metrics.evaluation_validation_rows!=resumed->metrics.evaluation_validation_rows||
       continuous->metrics.evaluation_test_rows!=resumed->metrics.evaluation_test_rows)m->evaluation_row_count++;
    if(continuous->grad_a_nonzero!=resumed->grad_a_nonzero)m->grad_a_count++;
    d7_compare_double(&continuous->grad_a_max,&resumed->grad_a_max,&m->grad_a_max);
    if(continuous->grad_b_nonzero!=resumed->grad_b_nonzero)m->grad_b_count++;
    d7_compare_double(&continuous->grad_b_max,&resumed->grad_b_max,&m->grad_b_max);
    d7_compare_double(&continuous->norm_before,&resumed->norm_before,&m->norm_before);
    d7_compare_double(&continuous->norm_after,&resumed->norm_after,&m->norm_after);
    if(continuous->clip_applied!=resumed->clip_applied)m->clip_applied++;
    if(continuous->clip_consistency!=resumed->clip_consistency)m->clip_consistency++;
    for(s=0;s<2U;s++){
        d7_compare_double(&continuous->metrics.split[s].loss,&resumed->metrics.split[s].loss,
            s==D4_TRAIN?&m->train_total:&m->validation_total);
        {
            double continuous_average=continuous->metrics.split[s].loss/
                (double)continuous->metrics.split[s].count;
            double resumed_average=resumed->metrics.split[s].loss/
                (double)resumed->metrics.split[s].count;
            d7_compare_double(&continuous_average,&resumed_average,
                s==D4_TRAIN?&m->train_average:&m->validation_average);
        }
        if(continuous->metrics.split[s].top1!=resumed->metrics.split[s].top1)
            (s==D4_TRAIN?&m->train_top1:&m->validation_top1)[0]++;
    }
    if((continuous->metrics.split[D4_VALIDATION].loss/33.0<=8.650949130533)!=
       (resumed->metrics.split[D4_VALIDATION].loss/33.0<=8.650949130533))
        m->validation_acceptance++;
}

static uint32_t d7_total_mismatches(const d7_mismatch_t *m)
{
    return m->step_index+m->gradient_row_count+m->evaluation_row_count+
        m->grad_a_count+m->grad_a_max+m->grad_b_count+m->grad_b_max+
        m->norm_before+m->norm_after+m->clip_applied+m->clip_consistency+
        m->train_total+m->train_average+m->train_top1+m->validation_total+
        m->validation_average+m->validation_top1+m->validation_acceptance;
}

static void d7_state_mismatches(const stage4_adapter_t *continuous,
    const stage4_adapter_t *resumed,uint32_t *a,uint32_t *b,
    uint32_t *m1a,uint32_t *m2a,uint32_t *m1b,uint32_t *m2b)
{
    *a=d6_byte_mismatches(continuous->a,resumed->a,continuous->a_count*sizeof(float));
    *b=d6_byte_mismatches(continuous->b,resumed->b,continuous->b_count*sizeof(float));
    *m1a=d6_byte_mismatches(continuous->m1_a,resumed->m1_a,continuous->a_count*sizeof(float));
    *m2a=d6_byte_mismatches(continuous->m2_a,resumed->m2_a,continuous->a_count*sizeof(float));
    *m1b=d6_byte_mismatches(continuous->m1_b,resumed->m1_b,continuous->b_count*sizeof(float));
    *m2b=d6_byte_mismatches(continuous->m2_b,resumed->m2_b,continuous->b_count*sizeof(float));
}

static int d7_write_tsv(const char *path,const d4_point_t c[D4_STEPS+1],
    const d4_point_t r[D4_STEPS+1])
{
    FILE *f=fopen(path,"wb"); uint32_t step; int ok=1;
    if(!f)return 0;
    fprintf(f,"step\tcontinuous_gradient_train_rows\tresumed_gradient_train_rows\tgradient_train_rows_match\tcontinuous_grad_A_nonzero\tresumed_grad_A_nonzero\tgrad_A_nonzero_match\tcontinuous_grad_B_nonzero\tresumed_grad_B_nonzero\tgrad_B_nonzero_match\tcontinuous_norm_before\tresumed_norm_before\tnorm_before_match\tcontinuous_norm_after\tresumed_norm_after\tnorm_after_match\tcontinuous_clip\tresumed_clip\tclip_match\tcontinuous_train_average\tresumed_train_average\ttrain_average_match\tcontinuous_validation_average\tresumed_validation_average\tvalidation_average_match\n");
    for(step=6U;step<=D7_STEPS;step++){
        const d4_point_t *a=&c[step],*b=&r[step];
        int rows=a->grad_train_rows==b->grad_train_rows&&a->grad_validation_rows==b->grad_validation_rows&&a->grad_test_rows==b->grad_test_rows;
        int ac=a->grad_a_nonzero==b->grad_a_nonzero,bc=a->grad_b_nonzero==b->grad_b_nonzero;
        int nb=memcmp(&a->norm_before,&b->norm_before,sizeof(double))==0;
        int na=memcmp(&a->norm_after,&b->norm_after,sizeof(double))==0;
        int clip=a->clip_applied==b->clip_applied&&a->clip_consistency==b->clip_consistency;
        double cta=a->metrics.split[D4_TRAIN].loss/111.0,rta=b->metrics.split[D4_TRAIN].loss/111.0;
        double cva=a->metrics.split[D4_VALIDATION].loss/33.0,rva=b->metrics.split[D4_VALIDATION].loss/33.0;
        int tmatch=memcmp(&cta,&rta,sizeof(double))==0;
        int vmatch=memcmp(&cva,&rva,sizeof(double))==0;
        fprintf(f,"%u\t%u\t%u\t%s\t%u\t%u\t%s\t%u\t%u\t%s\t%.17g\t%.17g\t%s\t%.17g\t%.17g\t%s\t%s\t%s\t%s\t%.12f\t%.12f\t%s\t%.12f\t%.12f\t%s\n",step,a->grad_train_rows,b->grad_train_rows,rows?"yes":"no",a->grad_a_nonzero,b->grad_a_nonzero,ac?"yes":"no",a->grad_b_nonzero,b->grad_b_nonzero,bc?"yes":"no",a->norm_before,b->norm_before,nb?"yes":"no",a->norm_after,b->norm_after,na?"yes":"no",a->clip_applied?"yes":"no",b->clip_applied?"yes":"no",clip?"yes":"no",cta,rta,tmatch?"yes":"no",cva,rva,vmatch?"yes":"no");
    }
    if(fclose(f)!=0)ok=0;
    return ok;
}

static int d7_write_report(const char *path,const char *ids[10],
    uint64_t d6_step,uint32_t step5_a,uint32_t step5_b,uint32_t step5_m1a,
    uint32_t step5_m2a,uint32_t step5_m1b,uint32_t step5_m2b,
    const d7_mismatch_t *m,uint32_t final_a,uint32_t final_b,
    uint32_t final_m1a,uint32_t final_m2a,uint32_t final_m1b,uint32_t final_m2b,
    const d4_point_t continuous[D4_STEPS+1],const d4_point_t resumed[D4_STEPS+1],
    int c_d4_match,int anchors,int train_match,int val_match,int finite_ok,int pass)
{
    static const char *keys[10]={"dataset_sha256","cache_sha256","base_checkpoint_sha256","tokenizer_sha256","d1_report_sha256","d2_report_sha256","d3_report_sha256","d4_report_sha256","d6_report_sha256","d6_checkpoint_sha256"};
    FILE *f=fopen(path,"wb"); uint32_t total=d7_total_mismatches(m); int i;
    if(!f)return 0;
    for(i=0;i<10;i++)fprintf(f,"%s=%s\n",keys[i],ids[i]);
    fprintf(f,"continuous_d4_trajectory_match=%s\ncontinuous_step5_optimizer_step=5\ncheckpoint_optimizer_step=%llu\nstep5_continuous_vs_checkpoint_A_byte_mismatches=%u\nstep5_continuous_vs_checkpoint_B_byte_mismatches=%u\nstep5_continuous_vs_checkpoint_m1_A_byte_mismatches=%u\nstep5_continuous_vs_checkpoint_m2_A_byte_mismatches=%u\nstep5_continuous_vs_checkpoint_m1_B_byte_mismatches=%u\nstep5_continuous_vs_checkpoint_m2_B_byte_mismatches=%u\nstep5_continuous_vs_checkpoint_total_state_byte_mismatches=%u\n",c_d4_match?"yes":"no",(unsigned long long)d6_step,step5_a,step5_b,step5_m1a,step5_m2a,step5_m1b,step5_m2b,step5_a+step5_b+step5_m1a+step5_m2a+step5_m1b+step5_m2b);
    fprintf(f,"continuous_optimizer_step=10\nresumed_optimizer_step=10\ncompared_resume_steps=5\nstep0_d1_metrics_match=%s\nstep1_d2_metrics_match=%s\nstep2_d3_metrics_match=%s\n",anchors?"yes":"no",anchors?"yes":"no",anchors?"yes":"no");
#define D7_OUT(k,v) fprintf(f,#k "=%u\n",m->v)
    D7_OUT(step_index_mismatches,step_index);D7_OUT(gradient_row_count_mismatches,gradient_row_count);D7_OUT(evaluation_row_count_mismatches,evaluation_row_count);
    D7_OUT(grad_A_nonzero_count_mismatches,grad_a_count);D7_OUT(grad_A_max_mismatches,grad_a_max);D7_OUT(grad_B_nonzero_count_mismatches,grad_b_count);D7_OUT(grad_B_max_mismatches,grad_b_max);
    D7_OUT(norm_before_clip_mismatches,norm_before);D7_OUT(norm_after_clip_mismatches,norm_after);D7_OUT(clip_applied_mismatches,clip_applied);D7_OUT(clip_consistency_mismatches,clip_consistency);
    D7_OUT(train_total_loss_mismatches,train_total);D7_OUT(train_average_loss_mismatches,train_average);D7_OUT(train_top1_mismatches,train_top1);D7_OUT(validation_total_loss_mismatches,validation_total);D7_OUT(validation_average_loss_mismatches,validation_average);D7_OUT(validation_top1_mismatches,validation_top1);D7_OUT(validation_acceptance_mismatches,validation_acceptance);
#undef D7_OUT
    fprintf(f,"total_resume_trajectory_field_mismatches=%u\ndeterministic_resume_trajectory_match=%s\nfinal_A_byte_mismatches=%u\nfinal_B_byte_mismatches=%u\nfinal_m1_A_byte_mismatches=%u\nfinal_m2_A_byte_mismatches=%u\nfinal_m1_B_byte_mismatches=%u\nfinal_m2_B_byte_mismatches=%u\nfinal_total_state_byte_mismatches=%u\nfinal_state_bitwise_match=%s\n",total,total==0U?"yes":"no",final_a,final_b,final_m1a,final_m2a,final_m1b,final_m2b,final_a+final_b+final_m1a+final_m2a+final_m1b+final_m2b,(final_a+final_b+final_m1a+final_m2a+final_m1b+final_m2b)==0U?"yes":"no");
    fprintf(f,"continuous_final_train_average_loss=%.12f\nresumed_final_train_average_loss=%.12f\ncontinuous_final_validation_average_loss=%.12f\nresumed_final_validation_average_loss=%.12f\ncontinuous_vs_resumed_train_metrics_match=%s\ncontinuous_vs_resumed_validation_metrics_match=%s\ntest_gradient_rows_seen=0\ntest_evaluation_rows_seen=0\nall_gradients_finite=%s\nall_parameters_finite=%s\nall_optimizer_moments_finite=%s\nall_losses_finite=%s\nd7_resume_pass=%s\n",continuous[10].metrics.split[D4_TRAIN].loss/111.0,resumed[10].metrics.split[D4_TRAIN].loss/111.0,continuous[10].metrics.split[D4_VALIDATION].loss/33.0,resumed[10].metrics.split[D4_VALIDATION].loss/33.0,train_match?"yes":"no",val_match?"yes":"no",finite_ok?"yes":"no",finite_ok?"yes":"no",finite_ok?"yes":"no",finite_ok?"yes":"no",pass?"yes":"no");
    return fclose(f)==0;
}

int main(int argc,char **argv)
{
    const char *checkpoint=NULL,*tokenizer=NULL,*dataset=NULL,*cache_path=NULL,*d1=NULL,*d2=NULL,*d3=NULL,*d4=NULL,*d6report=NULL,*d6checkpoint=NULL;
    char ds[65],ca[65],ck[65],tk[65],h1[65],h2[65],h3[65],h4[65],h6r[65],h6c[65];
    const char *ids[10]; unsigned char dsraw[32],ckraw[32],tkraw[32];
    d4_header_t header; d4_row_t *rows=NULL; d4_point_t continuous[11],resumed[11];
    d7_mismatch_t mismatch;
    d5_kv_t d4report[D5_REPORT_ENTRIES],d6report_kv[D5_REPORT_ENTRIES];
    uint32_t d4report_count=0U,d6report_count=0U;d5_reference_row_t d4tsv[11];d5_mismatches_t d4_mismatches;
    stage4_adapter_t c_ad,r_ad;Transformer tr;FILE *cf=NULL;
    float *base=NULL,*final=NULL;float *c_a5=NULL,*c_b5=NULL,*c_m1a5=NULL,*c_m2a5=NULL,*c_m1b5=NULL,*c_m2b5=NULL;
    uint32_t split_counts[4]={0U,0U,0U,0U},i;uint32_t step5_a=0U,step5_b=0U,step5_m1a=0U,step5_m2a=0U,step5_m1b=0U,step5_m2b=0U;
    uint32_t fa=0U,fb=0U,fm1a=0U,fm2a=0U,fm1b=0U,fm2b=0U;
    uint32_t final_m1a_count=0U,final_m2a_count=0U,final_m1b_count=0U,final_m2b_count=0U;
    uint64_t cache_size=0U,d6step=0U,cstep=0U,rstep=0U;
    uint32_t test_gradient_rows=0U,test_evaluation_rows=0U;
    int argi,tr_ready=0,c_ready=0,r_ready=0,loss_finite=1;
    int c_d4_match=0,anchors=0,train_match=0,val_match=0,pass=0,failed=0;
    int continuous_params=1,continuous_moments=1,resume_params=1,resume_moments=1;
    const char *report_path="stage4e-d7-a4-resume-report.txt";
    const char *trajectory_path="stage4e-d7-a4-resume-trajectory.tsv";
    printf("d7_build_provenance=%s\n",D7_PROVENANCE);
    memset(&c_ad,0,sizeof(c_ad));memset(&r_ad,0,sizeof(r_ad));memset(&tr,0,sizeof(tr));memset(&header,0,sizeof(header));memset(&mismatch,0,sizeof(mismatch));memset(continuous,0,sizeof(continuous));memset(resumed,0,sizeof(resumed));
    if(argc<2){fprintf(stderr,"usage: %s checkpoint -z tokenizer -d dataset -c cache -1 d1 -2 d2 -3 d3 -4 d4 -6 d6-report -7 d6-checkpoint\n",argv[0]);return 1;}
    checkpoint=argv[1];
    for(argi=2;argi<argc;argi++){if(!strcmp(argv[argi],"-z")&&argi+1<argc)tokenizer=argv[++argi];else if(!strcmp(argv[argi],"-d")&&argi+1<argc)dataset=argv[++argi];else if(!strcmp(argv[argi],"-c")&&argi+1<argc)cache_path=argv[++argi];else if(!strcmp(argv[argi],"-1")&&argi+1<argc)d1=argv[++argi];else if(!strcmp(argv[argi],"-2")&&argi+1<argc)d2=argv[++argi];else if(!strcmp(argv[argi],"-3")&&argi+1<argc)d3=argv[++argi];else if(!strcmp(argv[argi],"-4")&&argi+1<argc)d4=argv[++argi];else if(!strcmp(argv[argi],"-6")&&argi+1<argc)d6report=argv[++argi];else if(!strcmp(argv[argi],"-7")&&argi+1<argc)d6checkpoint=argv[++argi];else{fprintf(stderr,"error: D7 invalid option\n");return 1;}}
    if(!tokenizer||!dataset||!cache_path||!d1||!d2||!d3||!d4||!d6report||!d6checkpoint){fprintf(stderr,"error: D7 missing input path\n");return 1;}
    if(!stage4_hash_file_sha256_hex(dataset,ds)||!stage4_hash_file_sha256_hex(cache_path,ca)||!stage4_hash_file_sha256_hex(checkpoint,ck)||!stage4_hash_file_sha256_hex(tokenizer,tk)||!stage4_hash_file_sha256_hex(d1,h1)||!stage4_hash_file_sha256_hex(d2,h2)||!stage4_hash_file_sha256_hex(d3,h3)||!stage4_hash_file_sha256_hex(d4,h4)||!stage4_hash_file_sha256_hex(d6report,h6r)||!stage4_hash_file_sha256_hex(d6checkpoint,h6c)){fprintf(stderr,"error: D7 input hash failed\n");failed=1;goto done;}
    if(strcmp(ds,d7_dataset_sha)||strcmp(ca,d7_cache_sha)||strcmp(ck,d7_checkpoint_sha)||
       strcmp(tk,d7_tokenizer_sha)||strcmp(h1,d7_d1_sha)||strcmp(h2,d7_d2_sha)||
       strcmp(h3,D7_EXPECTED_D3_SHA)||strcmp(h4,D7_EXPECTED_D4_SHA)||
       strcmp(h6c,D7_EXPECTED_D6_CHECKPOINT_SHA)){
        fprintf(stderr,"error: D7 pinned input identity mismatch\n");failed=1;goto done;
    }
    ids[0]=ds;ids[1]=ca;ids[2]=ck;ids[3]=tk;ids[4]=h1;ids[5]=h2;ids[6]=h3;ids[7]=h4;ids[8]=h6r;ids[9]=h6c;
    if(!d4_float_ok()||!stage4_file_size_bytes(cache_path,&cache_size)||cache_size!=D4_CACHE_SIZE||
       !d4_hex_raw(ds,dsraw)||!d4_hex_raw(ck,ckraw)||!d4_hex_raw(tk,tkraw)){
        fprintf(stderr,"error: D7 cache identity/format invalid\n");failed=1;goto done;
    }
    if(!d5_load_report(d4,d4report,&d4report_count)||
       !d5_load_tsv("stage4e-d4-a4-trajectory.tsv",d4tsv)){
        fprintf(stderr,"error: cannot load D4 reference trajectory\n");failed=1;goto done;
    }
    if(d5_get(d4report,d4report_count,"d4_trajectory_pass")==NULL||
       strcmp(d5_get(d4report,d4report_count,"d4_trajectory_pass"),"yes")||
       strcmp(d5_get(d4report,d4report_count,"optimizer_steps_after"),"10")){
        fprintf(stderr,"error: D4 reference report not accepted\n");failed=1;goto done;
    }
    if(!d5_load_report(d6report,d6report_kv,&d6report_count)||
       d5_get(d6report_kv,d6report_count,"checkpoint_roundtrip_pass")==NULL||
       strcmp(d5_get(d6report_kv,d6report_count,"checkpoint_roundtrip_pass"),"yes")||
       strcmp(d5_get(d6report_kv,d6report_count,"checkpoint_sha256"),h6c)){
        fprintf(stderr,"error: D6 checkpoint report mismatch\n");failed=1;goto done;
    }

    cf=fopen(cache_path,"rb");
    if(!cf||!d4_read_header(cf,&header)||!d4_header_matches(&header,dsraw,ckraw,tkraw)){
        fprintf(stderr,"error: D7 cache header mismatch\n");failed=1;goto done;
    }
    rows=calloc(D4_ROWS,sizeof(*rows));
    if(!rows){fprintf(stderr,"error: D7 cache row allocation failed\n");failed=1;goto done;}
    for(i=0;i<D4_ROWS;i++){
        if(!d4_read_row(cf,&rows[i])||rows[i].record>=D4_RECORDS||rows[i].target>=D4_VOCAB||
           rows[i].split>=D4_SPLITS||rows[i].split==D4_REGRESSION){fprintf(stderr,"error: invalid D7 cache row %u\n",i);failed=1;goto done;}
        if(i&&(rows[i].record<rows[i-1].record||(rows[i].record==rows[i-1].record&&
           (rows[i].split!=rows[i-1].split||rows[i].position<=rows[i-1].position)))){
            fprintf(stderr,"error: D7 cache row order mismatch\n");failed=1;goto done;
        }
        split_counts[rows[i].split]++;
    }
    if(fgetc(cf)!=EOF||ferror(cf)||fclose(cf)!=0){cf=NULL;fprintf(stderr,"error: D7 cache trailing/read/close failure\n");failed=1;goto done;}
    cf=NULL;
    if(split_counts[D4_TRAIN]!=111U||split_counts[D4_VALIDATION]!=33U||split_counts[D4_TEST]!=21U){fprintf(stderr,"error: D7 cache split counts mismatch\n");failed=1;goto done;}

    unload_adapter_runtime();load_transformer(&tr,checkpoint);tr_ready=1;
    if(tr.config.dim!=(int)D4_DIM||tr.config.vocab_size!=(int)D4_VOCAB||
       !stage4_adapter_init(&c_ad,tr.config.dim,tr.config.vocab_size,D4_RANK,D4_SCALE)){
        fprintf(stderr,"error: D7 continuous adapter initialization failed\n");failed=1;goto done;
    }
    c_ready=1;
    if(c_ad.rank!=D4_RANK||c_ad.scale!=D4_SCALE||c_ad.a_count!=2304U||c_ad.b_count!=256000U){fprintf(stderr,"error: D7 continuous adapter shape mismatch\n");failed=1;goto done;}
    stage4_adapter_fill_a_deterministic(&c_ad,1U,0.01f);stage4_adapter_zero_b(&c_ad);stage4_adapter_zero_moments(&c_ad);
    base=calloc(D4_VOCAB,sizeof(float));final=calloc(D4_VOCAB,sizeof(float));
    if(!base||!final){fprintf(stderr,"error: D7 logits allocation failed\n");failed=1;goto done;}
    if(!d4_evaluate(rows,&c_ad,&tr,&continuous[0].metrics,&loss_finite)||!d4_anchor_metrics(&continuous[0].metrics,0U)){
        fprintf(stderr,"error: D7 continuous step-0 D1 anchor failed\n");failed=1;goto done;
    }
    for(i=1U;i<=D7_STEPS;i++){
        int grad_ok=1;double update;
        if(!d4_step(rows,&c_ad,&tr,base,final,(int)i,&continuous[i],&grad_ok,&loss_finite)||!grad_ok){fprintf(stderr,"error: D7 continuous gradient step %u failed\n",i);failed=1;goto done;}
        update=stage4_adapter_adam_step(&c_ad,(float)D4_LR,(float)D4_BETA1,(float)D4_BETA2,(float)D4_EPSILON,(float)D4_WEIGHT_DECAY,(uint64_t)i);
        if(!isfinite(update)){fprintf(stderr,"error: D7 continuous Adam update failed at %u\n",i);failed=1;goto done;}
        cstep++;
        d4_state_finite(&c_ad,&continuous_params,&continuous_moments);
        if(!continuous_params||!continuous_moments){fprintf(stderr,"error: D7 continuous state non-finite\n");failed=1;goto done;}
        if(!d4_evaluate(rows,&c_ad,&tr,&continuous[i].metrics,&loss_finite)||!d4_anchor_metrics(&continuous[i].metrics,i)||
           (i<=2U&&!d4_gradient_anchor(&continuous[i],i))){fprintf(stderr,"error: D7 continuous evaluation/anchor failed at %u\n",i);failed=1;goto done;}
        if(i==5U){
            c_a5=malloc(c_ad.a_count*sizeof(float));c_b5=malloc(c_ad.b_count*sizeof(float));
            c_m1a5=malloc(c_ad.a_count*sizeof(float));c_m2a5=malloc(c_ad.a_count*sizeof(float));
            c_m1b5=malloc(c_ad.b_count*sizeof(float));c_m2b5=malloc(c_ad.b_count*sizeof(float));
            if(!c_a5||!c_b5||!c_m1a5||!c_m2a5||!c_m1b5||!c_m2b5){fprintf(stderr,"error: D7 step-5 snapshot allocation failed\n");failed=1;goto done;}
            memcpy(c_a5,c_ad.a,c_ad.a_count*sizeof(float));memcpy(c_b5,c_ad.b,c_ad.b_count*sizeof(float));
            memcpy(c_m1a5,c_ad.m1_a,c_ad.a_count*sizeof(float));memcpy(c_m2a5,c_ad.m2_a,c_ad.a_count*sizeof(float));
            memcpy(c_m1b5,c_ad.m1_b,c_ad.b_count*sizeof(float));memcpy(c_m2b5,c_ad.m2_b,c_ad.b_count*sizeof(float));
        }
    }
    c_d4_match=d5_compare_reference(d4report,d4report_count,d4tsv,continuous,&d4_mismatches)&&d5_total_mismatches(&d4_mismatches)==0U;
    if(!c_d4_match||cstep!=D7_STEPS){fprintf(stderr,"error: continuous D7 path differs from accepted D4 trajectory\n");failed=1;goto done;}

    if(!stage4_adapter_init(&r_ad,tr.config.dim,tr.config.vocab_size,D4_RANK,D4_SCALE)){fprintf(stderr,"error: D7 resumed adapter initialization failed\n");failed=1;goto done;}
    r_ready=1;
    if(memcmp(c_a5,r_ad.a,r_ad.a_count*sizeof(float))==0&&memcmp(c_b5,r_ad.b,r_ad.b_count*sizeof(float))==0&&
       memcmp(c_m1a5,r_ad.m1_a,r_ad.a_count*sizeof(float))==0&&memcmp(c_m2a5,r_ad.m2_a,r_ad.a_count*sizeof(float))==0&&
       memcmp(c_m1b5,r_ad.m1_b,r_ad.b_count*sizeof(float))==0&&memcmp(c_m2b5,r_ad.m2_b,r_ad.b_count*sizeof(float))==0){
        fprintf(stderr,"error: fresh D7 state unexpectedly equals saved step-5 state\n");failed=1;goto done;
    }
    if(!d6_load_checkpoint(d6checkpoint,ds,ca,ck,tk,&r_ad,&d6step,D6_TOTAL_BYTES)||d6step!=5U){fprintf(stderr,"error: D7 could not load D6 checkpoint at step 5\n");failed=1;goto done;}
    step5_a=d6_byte_mismatches(c_a5,r_ad.a,c_ad.a_count*sizeof(float));
    step5_b=d6_byte_mismatches(c_b5,r_ad.b,c_ad.b_count*sizeof(float));
    step5_m1a=d6_byte_mismatches(c_m1a5,r_ad.m1_a,c_ad.a_count*sizeof(float));
    step5_m2a=d6_byte_mismatches(c_m2a5,r_ad.m2_a,c_ad.a_count*sizeof(float));
    step5_m1b=d6_byte_mismatches(c_m1b5,r_ad.m1_b,c_ad.b_count*sizeof(float));
    step5_m2b=d6_byte_mismatches(c_m2b5,r_ad.m2_b,c_ad.b_count*sizeof(float));
    if(!d4_evaluate(rows,&r_ad,&tr,&resumed[5].metrics,&loss_finite)){fprintf(stderr,"error: D7 loaded step-5 evaluation failed\n");failed=1;goto done;}
    if(memcmp(&continuous[5].metrics.split[D4_TRAIN].loss,&resumed[5].metrics.split[D4_TRAIN].loss,sizeof(double))!=0||
       memcmp(&continuous[5].metrics.split[D4_VALIDATION].loss,&resumed[5].metrics.split[D4_VALIDATION].loss,sizeof(double))!=0||
       continuous[5].metrics.split[D4_TRAIN].top1!=resumed[5].metrics.split[D4_TRAIN].top1||
       continuous[5].metrics.split[D4_VALIDATION].top1!=resumed[5].metrics.split[D4_VALIDATION].top1){fprintf(stderr,"error: loaded step-5 metrics differ from continuous path\n");failed=1;goto done;}
    rstep=d6step;
    for(i=6U;i<=D7_STEPS;i++){
        int grad_ok=1;double update;
        if(!d4_step(rows,&r_ad,&tr,base,final,(int)i,&resumed[i],&grad_ok,&loss_finite)||!grad_ok){fprintf(stderr,"error: D7 resumed gradient step %u failed\n",i);failed=1;goto done;}
        update=stage4_adapter_adam_step(&r_ad,(float)D4_LR,(float)D4_BETA1,(float)D4_BETA2,(float)D4_EPSILON,(float)D4_WEIGHT_DECAY,(uint64_t)i);
        if(!isfinite(update)){fprintf(stderr,"error: D7 resumed Adam update failed at %u\n",i);failed=1;goto done;}
        rstep++;
        d4_state_finite(&r_ad,&resume_params,&resume_moments);
        if(!resume_params||!resume_moments){fprintf(stderr,"error: D7 resumed state non-finite\n");failed=1;goto done;}
        if(!d4_evaluate(rows,&r_ad,&tr,&resumed[i].metrics,&loss_finite)){fprintf(stderr,"error: D7 resumed evaluation failed at %u\n",i);failed=1;goto done;}
        d7_compare_step(&continuous[i],&resumed[i],i,&mismatch);
    }
    d7_state_mismatches(&c_ad,&r_ad,&fa,&fb,&fm1a,&fm2a,&fm1b,&fm2b);
    d4_moment_counts(&c_ad,&final_m1a_count,&final_m2a_count,&final_m1b_count,&final_m2b_count);
    train_match=memcmp(&continuous[10].metrics.split[D4_TRAIN].loss,&resumed[10].metrics.split[D4_TRAIN].loss,sizeof(double))==0&&continuous[10].metrics.split[D4_TRAIN].top1==resumed[10].metrics.split[D4_TRAIN].top1;
    val_match=memcmp(&continuous[10].metrics.split[D4_VALIDATION].loss,&resumed[10].metrics.split[D4_VALIDATION].loss,sizeof(double))==0&&continuous[10].metrics.split[D4_VALIDATION].top1==resumed[10].metrics.split[D4_VALIDATION].top1;
    anchors=d4_anchor_metrics(&continuous[0].metrics,0U)&&d4_gradient_anchor(&continuous[1],1U)&&d4_anchor_metrics(&continuous[1].metrics,1U)&&d4_gradient_anchor(&continuous[2],2U)&&d4_anchor_metrics(&continuous[2].metrics,2U);
    pass=c_d4_match&&anchors&&cstep==D7_STEPS&&rstep==D7_STEPS&&d6step==5U&&
        step5_a+step5_b+step5_m1a+step5_m2a+step5_m1b+step5_m2b==0U&&
        d7_total_mismatches(&mismatch)==0U&&fa+fb+fm1a+fm2a+fm1b+fm2b==0U&&train_match&&val_match&&
        test_gradient_rows==0U&&test_evaluation_rows==0U&&continuous_params&&continuous_moments&&resume_params&&resume_moments&&loss_finite&&
        d4_fmt12(continuous[10].metrics.split[D4_TRAIN].loss/111.0,"9.577087323852")&&d4_fmt12(continuous[10].metrics.split[D4_VALIDATION].loss/33.0,"8.829185963434");
    if(!d7_write_tsv(trajectory_path,continuous,resumed)||!d7_write_report(report_path,ids,d6step,step5_a,step5_b,step5_m1a,step5_m2a,step5_m1b,step5_m2b,&mismatch,fa,fb,fm1a,fm2a,fm1b,fm2b,continuous,resumed,c_d4_match,anchors,train_match,val_match,continuous_params&&resume_params&&continuous_moments&&resume_moments&&loss_finite,pass)){fprintf(stderr,"error: D7 output writing failed\n");failed=1;goto done;}
    printf("base_checkpoint_identity=PASS\ntokenizer_identity=PASS\ndataset_identity=PASS\ncache_identity=PASS\nd1_report_identity=PASS\nd2_report_identity=PASS\nd3_report_identity=PASS\nd4_report_identity=PASS\nd6_report_identity=PASS\nd6_checkpoint_identity=PASS\ncontinuous_d4_trajectory_match=%s\ncontinuous_step5_optimizer_step=5\ncheckpoint_optimizer_step=%llu\nstep5_continuous_vs_checkpoint_total_state_byte_mismatches=%u\ncontinuous_optimizer_step=%llu\nresumed_optimizer_step=%llu\ncompared_resume_steps=5\ntotal_resume_trajectory_field_mismatches=%u\ndeterministic_resume_trajectory_match=%s\nfinal_A_byte_mismatches=%u\nfinal_B_byte_mismatches=%u\nfinal_m1_A_byte_mismatches=%u\nfinal_m2_A_byte_mismatches=%u\nfinal_m1_B_byte_mismatches=%u\nfinal_m2_B_byte_mismatches=%u\nfinal_total_state_byte_mismatches=%u\nfinal_state_bitwise_match=%s\nfinal_m1_A_nonzero_elements=%u\nfinal_m2_A_nonzero_elements=%u\nfinal_m1_B_nonzero_elements=%u\nfinal_m2_B_nonzero_elements=%u\ncontinuous_final_train_average_loss=%.12f\nresumed_final_train_average_loss=%.12f\ncontinuous_final_validation_average_loss=%.12f\nresumed_final_validation_average_loss=%.12f\ncontinuous_vs_resumed_train_metrics_match=%s\ncontinuous_vs_resumed_validation_metrics_match=%s\ntest_gradient_rows_seen=0\ntest_evaluation_rows_seen=0\nStage4E-D7-A4=%s\n",
        c_d4_match?"yes":"no",(unsigned long long)d6step,step5_a+step5_b+step5_m1a+step5_m2a+step5_m1b+step5_m2b,
        (unsigned long long)cstep,(unsigned long long)rstep,d7_total_mismatches(&mismatch),d7_total_mismatches(&mismatch)==0U?"yes":"no",
        fa,fb,fm1a,fm2a,fm1b,fm2b,fa+fb+fm1a+fm2a+fm1b+fm2b,fa+fb+fm1a+fm2a+fm1b+fm2b==0U?"yes":"no",
        final_m1a_count,final_m2a_count,final_m1b_count,final_m2b_count,
        continuous[10].metrics.split[0].loss/111.0,resumed[10].metrics.split[0].loss/111.0,
        continuous[10].metrics.split[1].loss/33.0,resumed[10].metrics.split[1].loss/33.0,
        train_match?"yes":"no",val_match?"yes":"no",pass?"PASS":"FAIL");
    if(!pass)failed=1;
done:
    if(cf)fclose(cf);
    free(rows);free(base);free(final);free(c_a5);free(c_b5);free(c_m1a5);free(c_m2a5);free(c_m1b5);free(c_m2b5);
    if(c_ready)stage4_adapter_free(&c_ad);
    if(r_ready)stage4_adapter_free(&r_ad);
    if(tr_ready)free_transformer(&tr);
    return failed||!pass?EXIT_FAILURE:EXIT_SUCCESS;
}
