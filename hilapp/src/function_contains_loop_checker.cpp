#include <sstream>
#include <iostream>
#include <string>

#include "myastvisitor.h"
#include "hilapp.h"


//////////////////////////////////////////////////////////////////////////////
/// An AST Visitor for checking if the function body contains a site loop
/// Logic: 
///   - Find if it contains X (X_index_type), which appears inside loops
///   - Find if it has Field[parity] -stmt.  This can appear without X in
///      statemets like  f[EVEN] = 1;  etc.
///
//////////////////////////////////////////////////////////////////////////////

class containsSiteLoopChecker : public GeneralVisitor, public RecursiveASTVisitor<containsSiteLoopChecker> {

public:
  using GeneralVisitor::GeneralVisitor;   // use general visitor constructor

  bool found_X;
  bool found_field_parity;
  bool found_field;
  bool searching_for_field;

  containsSiteLoopChecker(Rewriter &R,ASTContext *C,bool fieldsearch) : GeneralVisitor(R,C) {
    found_X = found_field_parity = false;
    searching_for_field = fieldsearch;
  }

  // bool VisitStmt(Stmt *s) { llvm::errs() << "In stmt\n"; return true; }

  bool VisitDeclRefExpr(DeclRefExpr * e) {
    /// if we visit X
    if (is_X_type(e)) {
      found_X = true;
      // llvm::errs() << "FOUND X index!\n";
      return false;  // we do not need to go further, do we?
    }
    if (is_field_expr(e)) {
      found_field = true;
      if (searching_for_field) return false; // stop
    }
    return true;
  }

  bool VisitCXXOperatorCallExpr(CXXOperatorCallExpr * OC) {
    if (is_field_parity_expr(OC)) {
      // llvm::errs() << "FOUND FIELD WITH PARITY! \n";
      found_field_parity = true;
      return false;
    }
    return true;
  }

};

///////////////////////////////////////////////////////////////////////////////////
/// And loop checker interface here
///////////////////////////////////////////////////////////////////////////////////

bool MyASTVisitor::does_function_contain_loop(FunctionDecl * f) {

  if (f->hasBody()) {
    containsSiteLoopChecker flc(TheRewriter,Context,false);
    flc.TraverseStmt(f->getBody());
    return (flc.found_X || flc.found_field_parity);
  }
  return false;
}


bool MyASTVisitor::does_expr_contain_field(Expr *E) {
  containsSiteLoopChecker flc(TheRewriter,Context,true);
  flc.TraverseStmt(E);
  return (flc.found_X || flc.found_field_parity || flc.found_field);
}