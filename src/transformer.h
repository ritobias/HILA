// -*- mode: c++ -*-
#ifndef TRANSFORMER_H
#define TRANSFORMER_H

#include <string>

#include "clang/AST/AST.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Frontend/ASTConsumers.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Tooling/Tooling.h"
#include "clang/Rewrite/Core/Rewriter.h"
//#include "llvm/Support/raw_ostream.h"



#undef NDEBUG
#include <assert.h>

enum class parity { none, even, odd, all, x };

struct loop_parity_struct {
  const Expr * expr;
  parity value;
  std::string text;
};


struct global_state {
  bool assert_loop_parity = false;
  std::string full_loop_text = "";
  unsigned in_func_template = 0;
  BinaryOperatorKind template_field_assignment_opcode;
  FunctionDecl * currentFunctionDecl = nullptr;
  struct location_struct {
    SourceLocation function;
    SourceLocation top;
    SourceLocation loop;
  } location;
};


// struct field_info;

struct field_ref {
  Expr * fullExpr;
  Expr * nameExpr;
  Expr * parityExpr;
  Expr * dirExpr;
  struct field_info * info;
  unsigned nameInd, parityInd;
  unsigned dirInd;
  bool is_changed;
};

struct dir_ptr {
  Expr * e;
  unsigned count;
};
  
struct field_info {
  std::string type;
  std::string old_name;
  std::string new_name;
  bool is_changed;   
  std::vector<dir_ptr> dir_list;
  std::vector<field_ref *> ref_list;

  field_info() {
    type = old_name = new_name = "";
    is_changed = false;
    dir_list = {};
    ref_list = {};
  }

  ~field_info() {
    dir_list.clear();
    ref_list.clear();
    type.clear();
    old_name.clear();
    new_name.clear();
  }
};

struct var_ref {
  DeclRefExpr *ref;
  unsigned ind;
  std::string assignop;
  bool is_assigned;
};

struct var_info {
  std::vector<var_ref> refs;
  VarDecl * decl;
  struct var_decl * var_declp;
  std::string type;
  std::string name;
  bool is_loop_local;
  bool is_reduction;
  bool is_assigned;
};


struct var_decl {
  VarDecl *decl;
  std::string name;
  std::string type;
  int scope;
};



#endif
