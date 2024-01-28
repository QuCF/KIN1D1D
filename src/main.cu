#include "../include/kin_outgoing.cuh"
#include "../include/kin_outgoing_diff.cuh"
// #include "../include/kin_zero_diff-SPACE.cuh"
using namespace std;


/**
 * ./kin nx nv wa 
*/
void kin_outgoing_form1(int argc, char *argv[]);


/**
 * To find out launch arguments, see description of the corresponding function.
*/
int main(int argc, char *argv[])
{
    kin_outgoing_form1(argc, argv);

    return 0;
}


void kin_outgoing_form1(int argc, char *argv[])
{
    using namespace std::complex_literals;
    Constants cc;
    string temp;
    double   wa;

    // --- input arguments ---
    uint id_arg = 0;
    uint32_t nx = stoi(string(argv[++id_arg]));
    uint32_t nv = stoi(string(argv[++id_arg]));
    stringstream(argv[++id_arg]) >> wa;

    // ---
    double Tref = 1.0e4 * cc.ev_; // in erg
    double den_ref = 1e12; // in cm-3

    uint Lx = 100;  // spatial size
    uint Lv = 4;  // velocity size
    double source_x0 = 50.;  // source position 
    double source_ds = 1.; // source width
    double diff = -0.00;
    string folder_to_save = "../../results/KIN1D1D-results/w12";

    // uint Lx = 50;  // spatial size
    // uint Lv = 4;  // velocity size
    // double source_x0 = 25.;  // source position 
    // double source_ds = 1.; // source width
    // double diff = -0.002;

    string id_prof = "flat";
    // string id_prof = "tanh2";
    // string id_prof = "exp";

    KWout_diff dd(nx, nv, Lx, Lv, Tref, den_ref, wa, 
        source_x0, source_ds, id_prof, diff, folder_to_save);
    // KWout dd(nx, nv, Lx, Lv, Tref, den_ref, wa, source_x0, source_ds, id_prof);
    // KWzero_diff dd(nx, nv, Lx, Lv, Tref, den_ref, wa, source_x0, source_ds, id_prof, diff);
    dd.init_device();

    dd.set_background_profiles();
    dd.save_xv_background_profs_to_hdf5();

    dd.form_rhs();
    dd.save_rhs_to_hdf5();

    dd.form_sparse_matrix();
    dd.save_matrices();

    dd.solve_system();

    dd.recheck_result();

    dd.save_result();
}


