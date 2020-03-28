#include <graph_types.hpp>

#include <openbabel/mol.h>
#include <openbabel/obconversion.h>
#include <openbabel/atom.h>
#include <openbabel/bond.h>

#include <fstream>

using namespace std;
using namespace OpenBabel;


#define EDGE_LABEL 0

types::Graph convert(const OBMol &mol)
{
  types::Graph host_graph;
  unsigned int num_atoms = mol.NumAtoms();
  for(unsigned int atomidx = 1; atomidx <= num_atoms; atomidx++) {
    OBAtom *atom = mol.GetAtom(atomidx);
    if(atom == 0) {
      cerr << "atom with id " << atomidx << " could not be retreived from molecule." << endl;
      abort();
    }
    host_graph.push_back(types::Vertex());
    host_graph.back().label = atom->GetAtomicNum();
    for(OBBondIterator itB = atom->BeginBonds(); itB != atom->EndBonds(); itB++) {
      host_graph.back().push(atomidx, (*itB)->GetEndAtom()->GetId(), EDGE_LABEL);
    } // for itB
  } // for atomidx

  return host_graph;
} // convert

int main(int argc, char **argv)
{
  ifstream in;
  ofstream out_ob;
  ofstream out_txt;

  string infilename = "nci3-filtered.sdf";
  string outfilename_ob = "test.out";
  string outfilename_txt = "nci_txt.dat";

  in.open(infilename.c_str(), std::ios::in);
  out_ob.open(outfilename_ob.c_str(), std::ios::out | std::ios::trunc);
  out_txt.open(outfilename_txt.c_str(), std::ios::out | std::ios::trunc);

  if(in.is_open() == false) {
    cerr << "could not open " << infilename << " file" << endl;
    return 1;
  }

  if(out_ob.is_open() == false) {
    cerr << "could not open " << outfilename_ob << endl;
    return 1;
  }

  OBConversion conv(&in, &out_ob);

  if(conv.SetInAndOutFormats("SD","SDF") == false) {
    std::cerr << "error while setting SDF format to openbabel" << std::endl;
  }


  OBMol mol;
  int objects_read = 0;

  while(conv.Read(&mol)) {
    types::Graph grph = convert(mol);
    out_txt << "t # " << objects_read << endl;
    grph.write(out_txt);

    if(objects_read % 500 == 0) cout << "processed object no: " << objects_read << endl;
    objects_read++;
  } // for(objid)

  return 0;
}


